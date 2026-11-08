"""Finite-difference reference for the 3D lid-driven cavity.

Independent of PINA: staggered (MAC) grid, central differences, explicit
Euler with Chorin projection and a DCT-based pressure Poisson solve, marched
to steady state. The lid at ``z = 1`` moves in ``x``; all other walls are
no-slip. Validated against Oueslati et al., C. R. Mecanique 339 (2011)
655-665, Table 1 (Re = 100: u_min = -0.2136 at 48^3).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.fft import dctn, idctn

log = logging.getLogger(__name__)


def smooth_lid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Regularised lid profile: 1 in the middle, 0 with zero slope at the edges."""
    return (16 * x * (1 - x) * y * (1 - y)) ** 2


def solve_cavity(
    reynolds: float,
    n: int,
    *,
    lid_speed: float = 1.0,
    smooth: bool = False,
    t_end: float = 30.0,
    tol: float = 1e-10,
) -> np.ndarray:
    """Steady velocity at the cell centres, shape ``(n, n, n, 3)``, index ``[x, y, z]``.

    Stops at ``t_end`` or once ``max |du/dt|`` drops below ``tol``.
    """
    nu = lid_speed / reynolds
    h = 1.0 / n
    dt = 0.9 * min(h * h / (6 * nu), 2 * nu / lid_speed**2, h / lid_speed)
    u = np.zeros((n + 1, n + 2, n + 2))
    v = np.zeros((n + 2, n + 1, n + 2))
    w = np.zeros((n + 2, n + 2, n + 1))
    x_faces = np.arange(n + 1) * h
    y_centres = (np.arange(n + 2) - 0.5) * h
    lid = lid_speed * (
        smooth_lid(x_faces[:, None], y_centres[None, :])
        if smooth
        else np.ones((n + 1, n + 2))
    )
    k = (2 * np.cos(np.pi * np.arange(n) / n) - 2) / h**2
    eigen = k[:, None, None] + k[None, :, None] + k[None, None, :]
    eigen[0, 0, 0] = 1.0

    def apply_walls() -> None:
        u[0], u[n] = 0.0, 0.0
        u[:, 0, :], u[:, n + 1, :] = -u[:, 1, :], -u[:, n, :]
        u[:, :, 0], u[:, :, n + 1] = -u[:, :, 1], 2 * lid - u[:, :, n]
        v[:, 0, :], v[:, n, :] = 0.0, 0.0
        v[0], v[n + 1] = -v[1], -v[n]
        v[:, :, 0], v[:, :, n + 1] = -v[:, :, 1], -v[:, :, n]
        w[:, :, 0], w[:, :, n] = 0.0, 0.0
        w[0], w[n + 1] = -w[1], -w[n]
        w[:, 0, :], w[:, n + 1, :] = -w[:, 1, :], -w[:, n, :]

    def laplacian(f: np.ndarray) -> np.ndarray:
        return (
            f[2:, 1:-1, 1:-1]
            + f[:-2, 1:-1, 1:-1]
            + f[1:-1, 2:, 1:-1]
            + f[1:-1, :-2, 1:-1]
            + f[1:-1, 1:-1, 2:]
            + f[1:-1, 1:-1, :-2]
            - 6 * f[1:-1, 1:-1, 1:-1]
        ) / h**2

    def gradient(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            (f[2:, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1]) / (2 * h),
            (f[1:-1, 2:, 1:-1] - f[1:-1, :-2, 1:-1]) / (2 * h),
            (f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, :-2]) / (2 * h),
        )

    def tendency() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Velocities interpolated onto the other components' faces.
        v_at_u = 0.25 * (
            v[1:n, 0:n, 1 : n + 1]
            + v[2 : n + 1, 0:n, 1 : n + 1]
            + v[1:n, 1 : n + 1, 1 : n + 1]
            + v[2 : n + 1, 1 : n + 1, 1 : n + 1]
        )
        w_at_u = 0.25 * (
            w[1:n, 1 : n + 1, 0:n]
            + w[2 : n + 1, 1 : n + 1, 0:n]
            + w[1:n, 1 : n + 1, 1 : n + 1]
            + w[2 : n + 1, 1 : n + 1, 1 : n + 1]
        )
        u_at_v = 0.25 * (
            u[0:n, 1:n, 1 : n + 1]
            + u[1 : n + 1, 1:n, 1 : n + 1]
            + u[0:n, 2 : n + 1, 1 : n + 1]
            + u[1 : n + 1, 2 : n + 1, 1 : n + 1]
        )
        w_at_v = 0.25 * (
            w[1 : n + 1, 1:n, 0:n]
            + w[1 : n + 1, 2 : n + 1, 0:n]
            + w[1 : n + 1, 1:n, 1 : n + 1]
            + w[1 : n + 1, 2 : n + 1, 1 : n + 1]
        )
        u_at_w = 0.25 * (
            u[0:n, 1 : n + 1, 1:n]
            + u[1 : n + 1, 1 : n + 1, 1:n]
            + u[0:n, 1 : n + 1, 2 : n + 1]
            + u[1 : n + 1, 1 : n + 1, 2 : n + 1]
        )
        v_at_w = 0.25 * (
            v[1 : n + 1, 0:n, 1:n]
            + v[1 : n + 1, 1 : n + 1, 1:n]
            + v[1 : n + 1, 0:n, 2 : n + 1]
            + v[1 : n + 1, 1 : n + 1, 2 : n + 1]
        )
        ux, uy, uz = gradient(u)
        vx, vy, vz = gradient(v)
        wx, wy, wz = gradient(w)
        du = -(u[1:-1, 1:-1, 1:-1] * ux + v_at_u * uy + w_at_u * uz) + nu * laplacian(u)
        dv = -(u_at_v * vx + v[1:-1, 1:-1, 1:-1] * vy + w_at_v * vz) + nu * laplacian(v)
        dw = -(u_at_w * wx + v_at_w * wy + w[1:-1, 1:-1, 1:-1] * wz) + nu * laplacian(w)
        return du, dv, dw

    def divergence() -> np.ndarray:
        return (
            (u[1 : n + 1, 1 : n + 1, 1 : n + 1] - u[0:n, 1 : n + 1, 1 : n + 1])
            + (v[1 : n + 1, 1 : n + 1, 1 : n + 1] - v[1 : n + 1, 0:n, 1 : n + 1])
            + (w[1 : n + 1, 1 : n + 1, 1 : n + 1] - w[1 : n + 1, 1 : n + 1, 0:n])
        ) / h

    apply_walls()
    steps = int(np.ceil(t_end / dt))
    for step in range(1, steps + 1):
        du, dv, dw = tendency()
        u_old = u[1:-1, 1:-1, 1:-1].copy()
        u[1:-1, 1:-1, 1:-1] += dt * du
        v[1:-1, 1:-1, 1:-1] += dt * dv
        w[1:-1, 1:-1, 1:-1] += dt * dw
        apply_walls()
        phi_hat = dctn(divergence() / dt, type=2, norm="ortho") / eigen
        phi_hat[0, 0, 0] = 0.0
        phi = idctn(phi_hat, type=2, norm="ortho")
        u[1:n, 1:-1, 1:-1] -= dt * (phi[1:] - phi[:-1]) / h
        v[1:-1, 1:n, 1:-1] -= dt * (phi[:, 1:] - phi[:, :-1]) / h
        w[1:-1, 1:-1, 1:n] -= dt * (phi[:, :, 1:] - phi[:, :, :-1]) / h
        apply_walls()
        if step % 200 == 0 or step == steps:
            change = float(np.abs(u[1:-1, 1:-1, 1:-1] - u_old).max() / dt)
            if not np.isfinite(change):
                raise RuntimeError(f"FD cavity diverged at step {step}")
            if change < tol:
                break
    log.info(
        "FD cavity Re=%g n=%d: %d steps, max|du/dt|=%.1e", reynolds, n, step, change
    )
    return np.stack(
        [
            0.5 * (u[1:, 1:-1, 1:-1] + u[:-1, 1:-1, 1:-1]),
            0.5 * (v[1:-1, 1:, 1:-1] + v[1:-1, :-1, 1:-1]),
            0.5 * (w[1:-1, 1:-1, 1:] + w[1:-1, 1:-1, :-1]),
        ],
        axis=-1,
    )


def cell_centres(n: int) -> np.ndarray:
    """Cell-centre coordinates in ``[x, y, z]`` order, shape ``(n**3, 3)``."""
    c = (np.arange(n) + 0.5) / n
    grid = np.meshgrid(c, c, c, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


def centerlines(vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``u_x(z)`` at ``x = y = 0.5`` and ``u_z(x)`` at ``y = z = 0.5`` (n even)."""
    n = vel.shape[0]
    mid = slice(n // 2 - 1, n // 2 + 1)
    s = (np.arange(n) + 0.5) / n
    return (
        s,
        vel[mid, mid, :, 0].mean(axis=(0, 1)),
        vel[:, mid, mid, 2].mean(axis=(1, 2)),
    )


def minimum(s: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """Minimum value and location, refined by a parabola through three samples."""
    i = int(np.argmin(f))
    if 0 < i < len(f) - 1:
        a, b, c = f[i - 1], f[i], f[i + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c)
        return float(b - 0.25 * (a - c) * d), float(s[i] + d * (s[1] - s[0]))
    return float(f[i]), float(s[i])
