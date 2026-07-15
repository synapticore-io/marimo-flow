"""Problem manager for creating PINA problems via a single API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import torch
from pina import Condition
from pina.domain import CartesianDomain
from pina.equation import Equation
from pina.equation.zoo import FixedValue
from pina.operator import grad, laplacian
from pina.problem import SpatialProblem, TimeDependentProblem
from pina.problem.zoo import SupervisedProblem


def _sin_product_ic(input_: Any, output_: Any) -> Any:
    """u(x,y,0) = sin(πx)·sin(πy) initial condition."""
    x = input_.extract(["x"])
    y = input_.extract(["y"])
    u = output_.extract(["u"])
    return u - torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def _build_time_dependent_domains(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    t_min: float,
    t_max: float,
) -> dict[str, CartesianDomain]:
    """Build standard 2D time-dependent domain dict (4 walls + IC + interior)."""
    return {
        "g1": CartesianDomain({"x": [x_min, x_max], "y": y_max, "t": [t_min, t_max]}),
        "g2": CartesianDomain({"x": [x_min, x_max], "y": y_min, "t": [t_min, t_max]}),
        "g3": CartesianDomain({"x": x_max, "y": [y_min, y_max], "t": [t_min, t_max]}),
        "g4": CartesianDomain({"x": x_min, "y": [y_min, y_max], "t": [t_min, t_max]}),
        "t0": CartesianDomain({"x": [x_min, x_max], "y": [y_min, y_max], "t": t_min}),
        "D": CartesianDomain(
            {"x": [x_min, x_max], "y": [y_min, y_max], "t": [t_min, t_max]}
        ),
    }


class ProblemManager:
    """Single entry point for creating problem classes/instances."""

    _PRESETS: dict[str, Callable[..., Any]] = {}

    @staticmethod
    def create_from_dataframe(
        df: Any,
        input_cols: list[str],
        output_cols: list[str],
    ) -> SupervisedProblem:
        """Create a SupervisedProblem from a DataFrame (Polars/Pandas).

        Args:
            df: Polars or Pandas DataFrame
            input_cols: List of column names for input features
            output_cols: List of column names for target values

        Returns:
            SupervisedProblem instance
        """
        if isinstance(df, pl.DataFrame):
            input_arr = df.select(input_cols).to_numpy()
            target_arr = df.select(output_cols).to_numpy()
        else:  # Pandas
            input_arr = df[input_cols].to_numpy()
            target_arr = df[output_cols].to_numpy()

        input_tensor = torch.from_numpy(input_arr).float()
        target_tensor = torch.from_numpy(target_arr).float()

        return SupervisedProblem(input_=input_tensor, output_=target_tensor)

    @staticmethod
    def create_supervised_problem(
        input_data: torch.Tensor,
        target_data: torch.Tensor,
    ) -> SupervisedProblem:
        """Create a supervised learning problem from data.

        Args:
            input_data: Input tensor
            target_data: Target tensor

        Returns:
            SupervisedProblem instance
        """
        return SupervisedProblem(input_=input_data, output_=target_data)

    @staticmethod
    def create_spatial_problem(
        output_variables: list[str],
        spatial_domain: CartesianDomain,
        domains: dict[str, CartesianDomain] | None = None,
        conditions: dict[str, Condition] | None = None,
    ) -> type[SpatialProblem]:
        """Create a spatial problem class.

        Args:
            output_variables: List of output variable names
            spatial_domain: Spatial domain definition
            domains: Optional dictionary of named domains
            conditions: Optional dictionary of conditions

        Returns:
            Problem class (not instance)
        """
        _ov, _sd = output_variables, spatial_domain
        _dom, _cond = domains or {}, conditions or {}

        class CustomSpatialProblem(SpatialProblem):
            output_variables = _ov
            spatial_domain = _sd
            domains = _dom
            conditions = _cond

        return CustomSpatialProblem

    @staticmethod
    def create_time_dependent_problem(
        output_variables: list[str],
        spatial_domain: CartesianDomain,
        temporal_domain: CartesianDomain,
        domains: dict[str, CartesianDomain] | None = None,
        conditions: dict[str, Condition] | None = None,
    ) -> type[TimeDependentProblem]:
        """Create a time-dependent problem class.

        Args:
            output_variables: List of output variable names
            spatial_domain: Spatial domain definition
            temporal_domain: Temporal domain definition
            domains: Optional dictionary of named domains
            conditions: Optional dictionary of conditions

        Returns:
            Problem class (not instance)
        """
        _ov, _sd, _td = output_variables, spatial_domain, temporal_domain
        _dom, _cond = domains or {}, conditions or {}

        class CustomTimeDependentProblem(TimeDependentProblem, SpatialProblem):
            output_variables = _ov
            spatial_domain = _sd
            temporal_domain = _td
            domains = _dom
            conditions = _cond

        return CustomTimeDependentProblem

    @staticmethod
    def create_poisson_problem(
        domain_bounds: dict[str, list[float]] | None = None,
        source_term: Callable | None = None,
    ) -> type[SpatialProblem]:
        """Create a Poisson problem class with configurable domain and source term.

        Args:
            domain_bounds: Domain bounds, e.g., {"x": [0, 1], "y": [0, 1]}
                         Defaults to unit square
            source_term: Source term function. Defaults to sin(pi*x)*sin(pi*y)

        Returns:
            Problem class (not instance)
        """
        if domain_bounds is None:
            domain_bounds = {"x": [0, 1], "y": [0, 1]}

        spatial_domain = CartesianDomain(domain_bounds)

        if source_term is None:

            def default_source(input_, output_):
                x = input_.extract(["x"])
                y = input_.extract(["y"])
                return -torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

            source_term = default_source

        def poisson_equation(input_, output_):
            lap_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
            f = source_term(input_, output_)
            return lap_u - f

        x_min, x_max = domain_bounds["x"]
        y_min, y_max = domain_bounds["y"]

        domains = {
            "g1": CartesianDomain({"x": [x_min, x_max], "y": y_max}),
            "g2": CartesianDomain({"x": [x_min, x_max], "y": y_min}),
            "g3": CartesianDomain({"x": x_max, "y": [y_min, y_max]}),
            "g4": CartesianDomain({"x": x_min, "y": [y_min, y_max]}),
            "D": spatial_domain,
        }

        conditions = {
            "g1": Condition(domain="g1", equation=FixedValue(0.0)),
            "g2": Condition(domain="g2", equation=FixedValue(0.0)),
            "g3": Condition(domain="g3", equation=FixedValue(0.0)),
            "g4": Condition(domain="g4", equation=FixedValue(0.0)),
            "D": Condition(domain="D", equation=Equation(poisson_equation)),
        }

        _sd, _dom, _cond = spatial_domain, domains, conditions

        class Poisson(SpatialProblem):
            output_variables = ["u"]
            spatial_domain = _sd
            domains = _dom
            conditions = _cond

        return Poisson

    @staticmethod
    def create_heat_problem(
        domain_bounds: dict[str, list[float]] | None = None,
        diffusivity: float = 0.01,
    ) -> type[TimeDependentProblem]:
        """Create a heat equation problem: du/dt = alpha * laplacian(u).

        Args:
            domain_bounds: Spatial and temporal bounds.
                         Defaults to {"x": [0,1], "y": [0,1], "t": [0,1]}
            diffusivity: Thermal diffusivity alpha. Defaults to 0.01

        Returns:
            Problem class (not instance)
        """
        if domain_bounds is None:
            domain_bounds = {"x": [0, 1], "y": [0, 1], "t": [0, 1]}

        spatial_vars = {k: v for k, v in domain_bounds.items() if k != "t"}
        spatial_domain = CartesianDomain(spatial_vars)
        temporal_domain = CartesianDomain({"t": domain_bounds["t"]})

        def heat_equation(input_, output_):
            du_dt = grad(output_, input_, components=["u"], d=["t"])
            lap_u = laplacian(output_, input_, components=["u"], d=list(spatial_vars))
            return du_dt - diffusivity * lap_u

        x_min, x_max = domain_bounds["x"]
        y_min, y_max = domain_bounds["y"]
        t_min, t_max = domain_bounds["t"]

        domains = _build_time_dependent_domains(
            x_min, x_max, y_min, y_max, t_min, t_max
        )
        conditions = {
            "g1": Condition(domain="g1", equation=FixedValue(0.0)),
            "g2": Condition(domain="g2", equation=FixedValue(0.0)),
            "g3": Condition(domain="g3", equation=FixedValue(0.0)),
            "g4": Condition(domain="g4", equation=FixedValue(0.0)),
            "t0": Condition(domain="t0", equation=Equation(_sin_product_ic)),
            "D": Condition(domain="D", equation=Equation(heat_equation)),
        }

        _sd, _td, _dom, _cond = spatial_domain, temporal_domain, domains, conditions

        class Heat(TimeDependentProblem, SpatialProblem):
            output_variables = ["u"]
            spatial_domain = _sd
            temporal_domain = _td
            domains = _dom
            conditions = _cond

        return Heat

    @staticmethod
    def create_burgers_problem(
        domain_bounds: dict[str, list[float]] | None = None,
        viscosity: float = 0.01 / torch.pi.item()
        if hasattr(torch.pi, "item")
        else 0.01 / 3.141592653589793,
        initial_condition: Callable | None = None,
    ) -> type[TimeDependentProblem]:
        """1D viscous Burgers equation: u_t + u*u_x = nu*u_xx.

        Default configuration matches the canonical PINN benchmark
        (Raissi et al. 2019): x in [-1, 1], t in [0, 1], u(x,0) = -sin(pi*x),
        u(-1,t) = u(1,t) = 0, nu = 0.01/pi.

        Args:
            domain_bounds: Defaults to {"x": [-1, 1], "t": [0, 1]}.
            viscosity: Kinematic viscosity nu. Defaults to 0.01/pi.
            initial_condition: Optional custom IC (input_, output_) -> residual.
                Defaults to u(x,0) = -sin(pi*x).

        Returns:
            Problem class (not instance).
        """
        if domain_bounds is None:
            domain_bounds = {"x": [-1.0, 1.0], "t": [0.0, 1.0]}

        spatial_domain = CartesianDomain({"x": domain_bounds["x"]})
        temporal_domain = CartesianDomain({"t": domain_bounds["t"]})
        x_min, x_max = domain_bounds["x"]
        t_min, t_max = domain_bounds["t"]

        def burgers_equation(input_, output_):
            u_t = grad(output_, input_, components=["u"], d=["t"])
            u_x = grad(output_, input_, components=["u"], d=["x"])
            u_xx = laplacian(output_, input_, components=["u"], d=["x"])
            u = output_.extract(["u"])
            return u_t + u * u_x - viscosity * u_xx

        if initial_condition is None:

            def _default_ic(input_, output_):
                x = input_.extract(["x"])
                u = output_.extract(["u"])
                return u - (-torch.sin(torch.pi * x))

            initial_condition = _default_ic

        domains = {
            "left": CartesianDomain({"x": x_min, "t": [t_min, t_max]}),
            "right": CartesianDomain({"x": x_max, "t": [t_min, t_max]}),
            "t0": CartesianDomain({"x": [x_min, x_max], "t": t_min}),
            "D": CartesianDomain({"x": [x_min, x_max], "t": [t_min, t_max]}),
        }
        conditions = {
            "left": Condition(domain="left", equation=FixedValue(0.0)),
            "right": Condition(domain="right", equation=FixedValue(0.0)),
            "t0": Condition(domain="t0", equation=Equation(initial_condition)),
            "D": Condition(domain="D", equation=Equation(burgers_equation)),
        }

        _sd, _td, _dom, _cond = spatial_domain, temporal_domain, domains, conditions

        class Burgers(TimeDependentProblem, SpatialProblem):
            output_variables = ["u"]
            spatial_domain = _sd
            temporal_domain = _td
            domains = _dom
            conditions = _cond

        return Burgers

    @staticmethod
    def create_allen_cahn_problem(
        domain_bounds: dict[str, list[float]] | None = None,
        epsilon: float = 0.01,
    ) -> type[TimeDependentProblem]:
        """1D Allen-Cahn equation: u_t = eps^2 * u_xx + u - u^3.

        Default: x in [-1, 1], t in [0, 1], homogeneous Dirichlet BCs,
        u(x,0) = x^2 * cos(pi*x) (canonical benchmark).

        Args:
            domain_bounds: Defaults to {"x": [-1, 1], "t": [0, 1]}.
            epsilon: Interface-width parameter. Defaults to 0.01.

        Returns:
            Problem class (not instance).
        """
        if domain_bounds is None:
            domain_bounds = {"x": [-1.0, 1.0], "t": [0.0, 1.0]}

        spatial_domain = CartesianDomain({"x": domain_bounds["x"]})
        temporal_domain = CartesianDomain({"t": domain_bounds["t"]})
        x_min, x_max = domain_bounds["x"]
        t_min, t_max = domain_bounds["t"]

        def allen_cahn_equation(input_, output_):
            u_t = grad(output_, input_, components=["u"], d=["t"])
            u_xx = laplacian(output_, input_, components=["u"], d=["x"])
            u = output_.extract(["u"])
            return u_t - epsilon**2 * u_xx - u + u**3

        def _ic(input_, output_):
            x = input_.extract(["x"])
            u = output_.extract(["u"])
            return u - x**2 * torch.cos(torch.pi * x)

        domains = {
            "left": CartesianDomain({"x": x_min, "t": [t_min, t_max]}),
            "right": CartesianDomain({"x": x_max, "t": [t_min, t_max]}),
            "t0": CartesianDomain({"x": [x_min, x_max], "t": t_min}),
            "D": CartesianDomain({"x": [x_min, x_max], "t": [t_min, t_max]}),
        }
        conditions = {
            "left": Condition(domain="left", equation=FixedValue(0.0)),
            "right": Condition(domain="right", equation=FixedValue(0.0)),
            "t0": Condition(domain="t0", equation=Equation(_ic)),
            "D": Condition(domain="D", equation=Equation(allen_cahn_equation)),
        }

        _sd, _td, _dom, _cond = spatial_domain, temporal_domain, domains, conditions

        class AllenCahn(TimeDependentProblem, SpatialProblem):
            output_variables = ["u"]
            spatial_domain = _sd
            temporal_domain = _td
            domains = _dom
            conditions = _cond

        return AllenCahn

    @staticmethod
    def create_advection_diffusion_problem(
        domain_bounds: dict[str, list[float]] | None = None,
        velocity: float = 1.0,
        diffusivity: float = 0.01,
    ) -> type[TimeDependentProblem]:
        """1D linear advection-diffusion: u_t + v*u_x = D*u_xx.

        Default: x in [0, 1], t in [0, 1], u(x,0) = sin(pi*x), homogeneous BCs.

        Args:
            domain_bounds: Defaults to {"x": [0, 1], "t": [0, 1]}.
            velocity: Advection speed v. Defaults to 1.0.
            diffusivity: Diffusion coefficient D. Defaults to 0.01.

        Returns:
            Problem class (not instance).
        """
        if domain_bounds is None:
            domain_bounds = {"x": [0.0, 1.0], "t": [0.0, 1.0]}

        spatial_domain = CartesianDomain({"x": domain_bounds["x"]})
        temporal_domain = CartesianDomain({"t": domain_bounds["t"]})
        x_min, x_max = domain_bounds["x"]
        t_min, t_max = domain_bounds["t"]

        def ad_equation(input_, output_):
            u_t = grad(output_, input_, components=["u"], d=["t"])
            u_x = grad(output_, input_, components=["u"], d=["x"])
            u_xx = laplacian(output_, input_, components=["u"], d=["x"])
            return u_t + velocity * u_x - diffusivity * u_xx

        def _ic(input_, output_):
            x = input_.extract(["x"])
            u = output_.extract(["u"])
            return u - torch.sin(torch.pi * x)

        domains = {
            "left": CartesianDomain({"x": x_min, "t": [t_min, t_max]}),
            "right": CartesianDomain({"x": x_max, "t": [t_min, t_max]}),
            "t0": CartesianDomain({"x": [x_min, x_max], "t": t_min}),
            "D": CartesianDomain({"x": [x_min, x_max], "t": [t_min, t_max]}),
        }
        conditions = {
            "left": Condition(domain="left", equation=FixedValue(0.0)),
            "right": Condition(domain="right", equation=FixedValue(0.0)),
            "t0": Condition(domain="t0", equation=Equation(_ic)),
            "D": Condition(domain="D", equation=Equation(ad_equation)),
        }

        _sd, _td, _dom, _cond = spatial_domain, temporal_domain, domains, conditions

        class AdvectionDiffusion(TimeDependentProblem, SpatialProblem):
            output_variables = ["u"]
            spatial_domain = _sd
            temporal_domain = _td
            domains = _dom
            conditions = _cond

        return AdvectionDiffusion

    @staticmethod
    def create_helmholtz_problem(
        domain_bounds: dict[str, list[float]] | None = None,
        wave_number: float = 1.0,
        source_term: Callable | None = None,
    ) -> type[SpatialProblem]:
        """2D Helmholtz equation: laplacian(u) + k^2 * u = f.

        Stationary (no time dimension). Default: unit square, k=1, f=0 (homogeneous).

        Args:
            domain_bounds: Defaults to {"x": [0, 1], "y": [0, 1]}.
            wave_number: Wave number k. Defaults to 1.0.
            source_term: Optional RHS source f(input_, output_). Defaults to 0.

        Returns:
            Problem class (not instance).
        """
        if domain_bounds is None:
            domain_bounds = {"x": [0.0, 1.0], "y": [0.0, 1.0]}

        spatial_domain = CartesianDomain(domain_bounds)
        x_min, x_max = domain_bounds["x"]
        y_min, y_max = domain_bounds["y"]

        def helmholtz_equation(input_, output_):
            lap_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
            u = output_.extract(["u"])
            f = source_term(input_, output_) if source_term else 0.0
            return lap_u + wave_number**2 * u - f

        domains = {
            "g1": CartesianDomain({"x": [x_min, x_max], "y": y_max}),
            "g2": CartesianDomain({"x": [x_min, x_max], "y": y_min}),
            "g3": CartesianDomain({"x": x_max, "y": [y_min, y_max]}),
            "g4": CartesianDomain({"x": x_min, "y": [y_min, y_max]}),
            "D": spatial_domain,
        }
        conditions = {
            "g1": Condition(domain="g1", equation=FixedValue(0.0)),
            "g2": Condition(domain="g2", equation=FixedValue(0.0)),
            "g3": Condition(domain="g3", equation=FixedValue(0.0)),
            "g4": Condition(domain="g4", equation=FixedValue(0.0)),
            "D": Condition(domain="D", equation=Equation(helmholtz_equation)),
        }

        _sd, _dom, _cond = spatial_domain, domains, conditions

        class Helmholtz(SpatialProblem):
            output_variables = ["u"]
            spatial_domain = _sd
            domains = _dom
            conditions = _cond

        return Helmholtz

    @staticmethod
    def create_wave_problem(
        domain_bounds: dict[str, list[float]] | None = None,
        wave_speed: float = 1.0,
    ) -> type[TimeDependentProblem]:
        """Create a wave equation problem: d2u/dt2 = c^2 * laplacian(u).

        Args:
            domain_bounds: Spatial and temporal bounds.
                         Defaults to {"x": [0,1], "y": [0,1], "t": [0,1]}
            wave_speed: Wave propagation speed c. Defaults to 1.0

        Returns:
            Problem class (not instance)
        """
        if domain_bounds is None:
            domain_bounds = {"x": [0, 1], "y": [0, 1], "t": [0, 1]}

        spatial_vars = {k: v for k, v in domain_bounds.items() if k != "t"}
        spatial_domain = CartesianDomain(spatial_vars)
        temporal_domain = CartesianDomain({"t": domain_bounds["t"]})

        def wave_equation(input_, output_):
            d2u_dt2 = laplacian(output_, input_, components=["u"], d=["t"])
            lap_u = laplacian(output_, input_, components=["u"], d=list(spatial_vars))
            return d2u_dt2 - wave_speed**2 * lap_u

        x_min, x_max = domain_bounds["x"]
        y_min, y_max = domain_bounds["y"]
        t_min, t_max = domain_bounds["t"]

        domains = _build_time_dependent_domains(
            x_min, x_max, y_min, y_max, t_min, t_max
        )

        def initial_velocity(input_, output_):
            du_dt = grad(output_, input_, components=["u"], d=["t"])
            return du_dt

        conditions = {
            "g1": Condition(domain="g1", equation=FixedValue(0.0)),
            "g2": Condition(domain="g2", equation=FixedValue(0.0)),
            "g3": Condition(domain="g3", equation=FixedValue(0.0)),
            "g4": Condition(domain="g4", equation=FixedValue(0.0)),
            "t0_u": Condition(domain="t0", equation=Equation(_sin_product_ic)),
            "t0_v": Condition(domain="t0", equation=Equation(initial_velocity)),
            "D": Condition(domain="D", equation=Equation(wave_equation)),
        }

        _sd, _td, _dom, _cond = spatial_domain, temporal_domain, domains, conditions

        class Wave(TimeDependentProblem, SpatialProblem):
            output_variables = ["u"]
            spatial_domain = _sd
            temporal_domain = _td
            domains = _dom
            conditions = _cond

        return Wave

    @classmethod
    def available(cls) -> tuple[str, ...]:
        """Return supported built-ins and registered presets."""
        builtin = (
            "poisson",
            "heat",
            "wave",
            "burgers",
            "allen_cahn",
            "advection_diffusion",
            "helmholtz",
            "spatial",
            "time_dependent",
            "supervised",
            "from_dataframe",
        )
        return tuple(sorted(set(builtin) | set(cls._PRESETS)))

    @classmethod
    def register(cls, kind: str, builder: Callable[..., Any]) -> None:
        """Register a custom problem builder under a kind name."""
        key = kind.strip().lower()
        cls._PRESETS[key] = builder

    @classmethod
    def create(cls, kind: str, **kwargs: Any) -> Any:
        """Create a problem by kind.

        Supported:
        - `poisson`: returns problem class
        - `heat`: returns problem class
        - `wave`: returns problem class
        - `spatial`: returns custom SpatialProblem class
        - `time_dependent`: returns custom TimeDependentProblem class
        - `supervised`: returns SupervisedProblem instance
        - `from_dataframe`: returns SupervisedProblem instance from table-like input
        - `<registered kind>`: calls custom registered builder

        Pass-through:
        - `problem`: if provided as instance/class, returned directly.
        """
        provided_problem = kwargs.pop("problem", None)
        if provided_problem is not None:
            return provided_problem

        key = kind.strip().lower()
        if key in cls._PRESETS:
            return cls._PRESETS[key](**kwargs)
        if key == "poisson":
            return cls.create_poisson_problem(**kwargs)
        if key == "heat":
            return cls.create_heat_problem(**kwargs)
        if key == "wave":
            return cls.create_wave_problem(**kwargs)
        if key == "burgers":
            return cls.create_burgers_problem(**kwargs)
        if key == "allen_cahn":
            return cls.create_allen_cahn_problem(**kwargs)
        if key == "advection_diffusion":
            return cls.create_advection_diffusion_problem(**kwargs)
        if key == "helmholtz":
            return cls.create_helmholtz_problem(**kwargs)
        if key == "spatial":
            return cls.create_spatial_problem(**kwargs)
        if key == "time_dependent":
            return cls.create_time_dependent_problem(**kwargs)
        if key == "supervised":
            return cls.create_supervised_problem(**kwargs)
        if key == "from_dataframe":
            return cls.create_from_dataframe(**kwargs)
        raise ValueError(
            f"Unknown problem kind '{kind}'. Available: {', '.join(cls.available())}"
        )
