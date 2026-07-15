"""1D heat-rod plant, ProblemSpec helper, and ROM surrogates for MPC demos.

* ``FiniteDifferenceHeatRod`` — explicit FTCS ground-truth plant.
* ``build_heat_rod_problem_spec`` — parametric PINN problem (``T(1)=u``).
* ``train_step_surrogate`` — optional MLP ROM trained on plant rollouts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from marimo_flow.agents.schemas import (
    ConditionSpec,
    ControlParameterSpec,
    DerivativeSpec,
    EquationSpec,
    ProblemSpec,
    SubdomainSpec,
)

PredictStepFn = Callable[[float, float], float]
SurrogateFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class FiniteDifferenceHeatRod:
    """Explicit FTCS stepper for ``T_t = alpha T_xx`` on x in [0, 1].

    Left boundary is fixed at 0; the right boundary temperature is the
    scalar control input ``u``.
    """

    nx: int = 51
    alpha: float = 0.05
    dt: float = 0.002
    T: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.nx < 3:
            raise ValueError("nx must be >= 3")
        self.dx = 1.0 / float(self.nx - 1)
        self._r = self.alpha * self.dt / (self.dx**2)
        if self._r > 0.5:
            raise ValueError(
                f"FTCS unstable: alpha*dt/dx^2={self._r:.4f} > 0.5; reduce dt or alpha"
            )
        self.T = np.zeros(self.nx, dtype=np.float64)

    @property
    def center_index(self) -> int:
        return self.nx // 2

    def center(self) -> float:
        return float(self.T[self.center_index])

    def mean(self) -> float:
        """Spatial mean — scalar MPC observation that responds in one step."""
        return float(self.T.mean())

    def set_uniform(self, value: float) -> None:
        self.T[:] = float(value)

    def set_sinusoidal(self, amplitude: float) -> None:
        x = np.linspace(0.0, 1.0, self.nx)
        self.T[:] = float(amplitude) * np.sin(np.pi * x)

    def randomize(self, rng: np.random.Generator) -> None:
        mode = rng.choice(["uniform", "sinusoidal"])
        amp = float(rng.uniform(0.0, 0.4))
        if mode == "uniform":
            self.set_uniform(amp)
        else:
            self.set_sinusoidal(amp)

    def step(self, u: float) -> float:
        """Advance one time step; return the new spatial mean."""
        old = self.T.copy()
        for i in range(1, self.nx - 1):
            self.T[i] = old[i] + self._r * (old[i - 1] - 2.0 * old[i] + old[i + 1])
        self.T[0] = 0.0
        self.T[-1] = float(u)
        return self.mean()


def generate_step_dataset(
    plant: FiniteDifferenceHeatRod,
    n_samples: int,
    rng: np.random.Generator,
    *,
    u_low: float = 0.0,
    u_high: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``(T_mean, u) -> T_mean_next`` pairs from the plant."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    inputs = np.zeros((n_samples, 2), dtype=np.float64)
    targets = np.zeros((n_samples, 1), dtype=np.float64)
    for i in range(n_samples):
        if i % 32 == 0:
            plant.randomize(rng)
        t_mean = plant.mean()
        u = float(rng.uniform(u_low, u_high))
        inputs[i, 0] = t_mean
        inputs[i, 1] = u
        targets[i, 0] = plant.step(u)
    return inputs, targets


def train_step_surrogate(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    hidden: tuple[int, ...] = (32, 32),
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> PredictStepFn:
    """Fit a small MLP on plant rollouts; return a numpy predict callable."""
    if inputs.ndim != 2 or inputs.shape[1] != 2:
        raise ValueError("inputs must have shape (n, 2)")
    if targets.ndim != 2 or targets.shape[1] != 1:
        raise ValueError("targets must have shape (n, 1)")
    if len(inputs) == 0:
        raise ValueError("inputs must be non-empty")

    x_mean = inputs.mean(axis=0)
    x_std = inputs.std(axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    y_mean = float(targets.mean())
    y_std = float(targets.std())
    if y_std < 1e-6:
        y_std = 1.0

    x_norm = (inputs - x_mean) / x_std
    y_norm = (targets - y_mean) / y_std

    layers: list[nn.Module] = []
    in_dim = 2
    for width in hidden:
        layers.extend([nn.Linear(in_dim, width), nn.Tanh()])
        in_dim = width
    layers.append(nn.Linear(in_dim, 1))
    model = nn.Sequential(*layers)

    x_t = torch.tensor(x_norm, dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(x_t)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x_t[idx])
            loss = loss_fn(pred, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()

    def predict(t_mean: float, u: float) -> float:
        raw = np.array([[t_mean, u]], dtype=np.float64)
        normed = (raw - x_mean) / x_std
        with torch.no_grad():
            out = model(torch.tensor(normed, dtype=torch.float32))
        return float(out.item() * y_std + y_mean)

    return predict


def make_rollout_surrogate(predict_step: PredictStepFn) -> SurrogateFn:
    """Chain one-step predictions over the MPC horizon."""

    def surrogate(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        horizon = len(controls)
        traj = np.zeros((horizon, state.shape[0]), dtype=np.float64)
        x = float(state[0])
        for i in range(horizon):
            x = predict_step(x, float(controls[i, 0]))
            traj[i, 0] = x
        return traj

    return surrogate


def make_plant_stepper(plant: FiniteDifferenceHeatRod) -> SurrogateFn:
    """Advance the full field; ignores ``state`` (plant memory is authoritative)."""

    def step(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        _ = state
        horizon = len(controls)
        traj = np.zeros((horizon, 1), dtype=np.float64)
        for i in range(horizon):
            traj[i, 0] = plant.step(float(controls[i, 0]))
        return traj

    return step


def build_heat_rod_problem_spec(
    *,
    alpha: float = 0.08,
    u_low: float = 0.0,
    u_high: float = 1.0,
) -> ProblemSpec:
    """1D heat equation with controllable right-boundary temperature ``u``."""
    ambient = {"x": [0.0, 1.0], "t": [0.0, 1.0], "u": [u_low, u_high]}
    return ProblemSpec(
        name="heat_rod_1d_controlled",
        output_variables=["T"],
        domain_bounds={"x": [0.0, 1.0], "t": [0.0, 1.0]},
        control_parameters=[
            ControlParameterSpec(name="u", low=u_low, high=u_high),
        ],
        subdomains=[
            SubdomainSpec(name="D", bounds=ambient),
            SubdomainSpec(
                name="left", bounds={"x": 0.0, "t": [0.0, 1.0], "u": [u_low, u_high]}
            ),
            SubdomainSpec(
                name="right",
                bounds={"x": 1.0, "t": [0.0, 1.0], "u": [u_low, u_high]},
            ),
            SubdomainSpec(
                name="t0", bounds={"x": [0.0, 1.0], "t": 0.0, "u": [u_low, u_high]}
            ),
        ],
        equations=[
            EquationSpec(
                name="heat",
                form="T_t - alpha*T_xx",
                outputs=["T"],
                derivatives=[
                    DerivativeSpec(name="T_t", field="T", wrt=["t"]),
                    DerivativeSpec(name="T_xx", field="T", wrt=["x", "x"]),
                ],
                parameters={"alpha": alpha},
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="heat"),
            ConditionSpec(subdomain="left", kind="fixed_value", value=0.0),
            ConditionSpec(
                subdomain="right",
                kind="parametric_dirichlet",
                parameter_name="u",
            ),
            ConditionSpec(subdomain="t0", kind="fixed_value", value=0.0),
        ],
    )
