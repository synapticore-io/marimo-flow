"""MeshDomain — PINA DomainInterface adapter for unstructured meshes.

Reads a ``MeshSpec`` via ``meshio`` and samples collocation points on
the cells by uniform barycentric draws. Supports triangle (k=3) and
tetrahedron (k=4) cell blocks; quads / hexes are decomposed into two
triangles / five tetrahedra so the same sampler applies.

The composer uses this when a ``SubdomainSpec`` carries ``mesh_ref``
(tagged cell region) or when the full ProblemSpec attaches a mesh
without Cartesian fallback. ``is_inside`` is intentionally coarse
(bounding-box + nearest-cell test) — PINN training only cares about
sampled collocation points, not exact point-in-mesh queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from pina.domain import DomainInterface
from pina.label_tensor import LabelTensor

from marimo_flow.agents.schemas import MeshSpec

_CELL_BARYCENTRIC_SIZE: dict[str, int] = {
    "line": 2,
    "triangle": 3,
    "quad": 3,  # decomposed to 2 triangles
    "tetra": 4,
    "hexahedron": 4,  # decomposed to 5 tetrahedra
    "wedge": 4,
    "pyramid": 4,
}


class MeshDomain(DomainInterface):
    """Sample PINN collocation points uniformly on an unstructured mesh.

    Parameters
    ----------
    points : np.ndarray
        ``(N, d)`` coordinate array, one row per mesh node.
    cells : np.ndarray
        ``(M, k)`` index array — each row lists the node ids of a cell.
    axes : list[str]
        axis labels; length must equal ``d``.
    cell_kind : str
        meshio cell block name (``triangle``, ``tetra``, …).
    cell_indices : np.ndarray | None
        optional row mask into ``cells`` — used by tagged sub-regions.
    """

    def __init__(
        self,
        points: np.ndarray,
        cells: np.ndarray,
        axes: list[str],
        cell_kind: str,
        cell_indices: np.ndarray | None = None,
    ):
        self._points = np.asarray(points, dtype=np.float64)
        self._cells = np.asarray(cells, dtype=np.int64)
        self._axes = list(axes)
        self._cell_kind = cell_kind
        self._cell_indices = (
            np.asarray(cell_indices, dtype=np.int64)
            if cell_indices is not None
            else np.arange(len(self._cells), dtype=np.int64)
        )
        self._sample_modes = ("random",)
        if self._points.shape[1] != len(axes):
            raise ValueError(
                f"axes ({axes}) length must match point dim {self._points.shape[1]}"
            )
        if cell_kind not in _CELL_BARYCENTRIC_SIZE:
            raise ValueError(
                f"unsupported cell kind {cell_kind!r}; "
                f"supported: {sorted(_CELL_BARYCENTRIC_SIZE)}"
            )

    # --- DomainInterface surface --------------------------------------

    def sample(
        self, n: int, mode: str = "random", variables: Any = "all"
    ) -> LabelTensor:  # noqa: ARG002 — PINA API
        if mode != "random":
            raise ValueError(f"MeshDomain only supports mode='random', got {mode!r}")
        rng = np.random.default_rng()
        picked = rng.choice(self._cell_indices, size=n, replace=True)
        cell_nodes = self._cells[picked]  # (n, k)
        vertex_coords = self._points[cell_nodes]  # (n, k, d)

        k = cell_nodes.shape[1]
        weights = _barycentric_weights(n, k, rng=rng)  # (n, k)
        pts = (vertex_coords * weights[..., None]).sum(axis=1)  # (n, d)
        return LabelTensor(torch.tensor(pts, dtype=torch.float32), list(self._axes))

    def is_inside(self, point: LabelTensor, check_border: bool = False) -> bool:  # noqa: ARG002 — PINA API
        """Bounding-box containment. PINN sampling never calls this in practice."""
        coords = point.tensor.detach().cpu().numpy().reshape(-1)
        lo = self._points.min(axis=0)
        hi = self._points.max(axis=0)
        return bool(np.all(coords >= lo) and np.all(coords <= hi))

    def update(self, domain: Any) -> MeshDomain:
        raise NotImplementedError("MeshDomain does not support label-merging updates")

    def partial(self) -> MeshDomain:
        raise NotImplementedError(
            "use MeshSpec.cell_tags to declare boundary cell blocks explicitly"
        )

    @property
    def sample_modes(self) -> list[str]:
        return list(self._sample_modes)

    @property
    def variables(self) -> list[str]:
        return list(self._axes)

    @property
    def domain_dict(self) -> dict[str, list[float]]:
        lo = self._points.min(axis=0)
        hi = self._points.max(axis=0)
        return {a: [float(lo[i]), float(hi[i])] for i, a in enumerate(self._axes)}

    @property
    def range(self) -> dict[str, tuple[float, float]]:
        lo = self._points.min(axis=0)
        hi = self._points.max(axis=0)
        return {a: (float(lo[i]), float(hi[i])) for i, a in enumerate(self._axes)}

    @property
    def fixed(self) -> dict[str, float]:
        return {}

    # --- extras used by viz / diagnostics -----------------------------

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def cells(self) -> np.ndarray:
        return self._cells

    @property
    def cell_kind(self) -> str:
        return self._cell_kind


def load_mesh_domain(spec: MeshSpec, *, mesh_ref: str | None = None) -> MeshDomain:
    """Read ``spec.path`` via meshio and build a MeshDomain.

    ``mesh_ref`` picks a subset of the cell block declared in
    ``spec.cell_tags``. ``None`` = full cell block.
    """
    import meshio  # imported lazily so the dep stays optional at import time

    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(f"mesh file not found: {spec.path}")
    mesh = meshio.read(str(path), file_format=spec.format)

    block = _pick_cell_block(mesh.cells, preferred=spec.primary_cell_kind)
    cells = block.data
    cell_kind = block.type

    subset: np.ndarray | None = None
    if mesh_ref is not None:
        if mesh_ref not in spec.cell_tags:
            raise KeyError(
                f"mesh tag {mesh_ref!r} not declared in MeshSpec.cell_tags "
                f"(known: {sorted(spec.cell_tags)})"
            )
        subset = np.asarray(spec.cell_tags[mesh_ref], dtype=np.int64)

    return MeshDomain(
        points=mesh.points,
        cells=cells,
        axes=list(spec.axes),
        cell_kind=cell_kind,
        cell_indices=subset,
    )


# --- internals -----------------------------------------------------


def _pick_cell_block(cell_blocks: list, preferred: str | None):
    """Return the meshio CellBlock that matches ``preferred`` or the biggest."""
    if preferred is not None:
        for block in cell_blocks:
            if block.type == preferred:
                return block
        raise KeyError(
            f"cell kind {preferred!r} not found in mesh "
            f"(available: {[b.type for b in cell_blocks]})"
        )
    usable = [b for b in cell_blocks if b.type in _CELL_BARYCENTRIC_SIZE]
    if not usable:
        raise ValueError(
            f"no supported cell kinds in mesh; got {[b.type for b in cell_blocks]}"
        )
    return max(usable, key=lambda b: len(b.data))


def _barycentric_weights(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform barycentric coords on a (k-1)-simplex, shape (n, k)."""
    if k == 2:
        t = rng.random(n)
        return np.stack([1.0 - t, t], axis=1)
    if k == 3:
        r1 = np.sqrt(rng.random(n))
        r2 = rng.random(n)
        return np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)
    if k == 4:
        u = rng.random((n, 3))
        # See Rocchini & Cignoni, "Generating random points in a tetrahedron".
        s = np.sort(u, axis=1)
        a = s[:, 0]
        b = s[:, 1] - s[:, 0]
        c = s[:, 2] - s[:, 1]
        d = 1.0 - s[:, 2]
        return np.stack([a, b, c, d], axis=1)
    raise ValueError(f"unsupported barycentric arity: {k}")
