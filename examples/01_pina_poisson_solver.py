# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # Pulls pina-mathlab, torch, plotly, ... Declared explicitly so marimo's
#     # sandbox does not auto-install the unrelated PyPI package "pina".
#     "marimo-flow>=0.4.5",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import mlflow
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import torch
    import torch.nn as nn
    from lightning.pytorch.loggers import MLFlowLogger
    from pina.label_tensor import LabelTensor

    from marimo_flow.core import (
        FoundationModelAdapter,
        ProblemManager,
        SolverManager,
        build_optuna_history_figure,
        build_optuna_parallel_figure,
        build_optuna_param_importance_figure,
        build_trials_scatter_chart,
        create_model_for_problem,
        study_trials_dataframe,
        train_solver,
    )

    mlflow.set_tracking_uri("sqlite:///data/mlflow/db/mlflow.db")

    PROBLEM_KINDS = {
        "Poisson": "poisson",
        "Heat Equation": "heat",
        "Wave Equation": "wave",
    }
    ACTIVATIONS = {"Tanh": nn.Tanh, "SiLU": nn.SiLU, "Softplus": nn.Softplus}
    # `marimo export html 01_....py -- --epochs 50 --trials 2 --trial-epochs 20`
    # shrinks the run for CI without touching the code.
    CLI = mo.cli_args()
    return (
        ACTIVATIONS,
        CLI,
        FoundationModelAdapter,
        LabelTensor,
        MLFlowLogger,
        PROBLEM_KINDS,
        ProblemManager,
        SolverManager,
        build_optuna_history_figure,
        build_optuna_parallel_figure,
        build_optuna_param_importance_figure,
        build_trials_scatter_chart,
        create_model_for_problem,
        go,
        mlflow,
        mo,
        np,
        pl,
        study_trials_dataframe,
        torch,
        train_solver,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # PINA Lab — Physics-Informed Neural Networks

        Select a PDE, configure the network architecture and press **Train**.
        The notebook trains once with the defaults on load, so it also runs
        headless (`marimo export`, `python 01_pina_poisson_solver.py`).
        Results are cached on disk (`mo.persistent_cache`): re-opening the
        notebook or returning to a previous configuration skips training.
        All fresh runs are tracked in MLflow.
        """
    )
    return


@app.cell
def _(ACTIVATIONS, CLI, PROBLEM_KINDS, mo):
    train_form = mo.ui.batch(
        mo.md(
            """
            {problem} {solver} {backbone} {model_id}

            {layer_size} {num_layers} {activation} {freeze}

            {lr} {epochs}
            """
        ),
        {
            "problem": mo.ui.dropdown(
                options=list(PROBLEM_KINDS), value="Poisson", label="Problem"
            ),
            "solver": mo.ui.dropdown(
                options=["PINN", "SAPINN"], value="PINN", label="Solver"
            ),
            "backbone": mo.ui.dropdown(
                options=["baseline", "foundation"], value="baseline", label="Backbone"
            ),
            "model_id": mo.ui.text(
                value="polymathic-ai/walrus", label="HuggingFace model id"
            ),
            "layer_size": mo.ui.slider(16, 128, value=64, step=16, label="Layer size"),
            "num_layers": mo.ui.slider(2, 5, value=3, step=1, label="Layers"),
            "activation": mo.ui.dropdown(
                options=list(ACTIVATIONS), value="Tanh", label="Activation"
            ),
            "freeze": mo.ui.checkbox(value=True, label="Freeze foundation backbone"),
            "lr": mo.ui.number(value=1e-3, label="Learning rate"),
            "epochs": mo.ui.slider(
                50, 3000, value=int(CLI.get("epochs", 1000)), step=50, label="Epochs"
            ),
        },
    ).form(submit_button_label="Train", bordered=True)
    train_form
    return (train_form,)


@app.cell
def _(train_form):
    # Defaults until the form is submitted, so the notebook trains on load.
    train_cfg = train_form.value or train_form.element.value
    return (train_cfg,)


@app.cell
def _(
    ACTIVATIONS,
    FoundationModelAdapter,
    PROBLEM_KINDS,
    ProblemManager,
    SolverManager,
    create_model_for_problem,
):
    def build_solver(cfg: dict):
        problem = ProblemManager.create(PROBLEM_KINDS[cfg["problem"]])()
        if cfg["backbone"] == "foundation":
            model = FoundationModelAdapter(
                checkpoint=cfg["model_id"].strip() or "polymathic-ai/walrus",
                input_dimensions=len(problem.input_variables),
                freeze_backbone=cfg["freeze"],
                out_labels=tuple(problem.output_variables),
            )
        else:
            model = create_model_for_problem(
                problem,
                layers=[cfg["layer_size"]] * cfg["num_layers"],
                activation=ACTIVATIONS[cfg["activation"]],
            )
        solver = SolverManager.create(
            "sapinn" if cfg["solver"] == "SAPINN" else "pinn",
            problem=problem,
            model=model,
            learning_rate=cfg["lr"],
        )
        return problem, model, solver

    return (build_solver,)


@app.cell
def _(MLFlowLogger, build_solver, mlflow, mo, torch, train_solver):
    @mo.persistent_cache
    def train_run(cfg: dict) -> tuple[dict, dict, float]:
        """Train once per configuration; returns weights, final metrics, seconds."""
        import time

        torch.manual_seed(0)
        _, model, solver = build_solver(cfg)
        experiment = f"pina-{cfg['problem'].lower().replace(' ', '-')}"
        mlflow.set_experiment(experiment)
        logger = MLFlowLogger(
            experiment_name=experiment,
            tracking_uri=mlflow.get_tracking_uri(),
            log_model=False,
            run_name=f"{cfg['backbone']}-{cfg['solver']}-{cfg['epochs']}ep",
        )
        logger.log_hyperparams(cfg)
        start = time.perf_counter()
        trainer = train_solver(solver, max_epochs=cfg["epochs"], logger=logger)
        seconds = time.perf_counter() - start
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        return state, metrics, seconds

    return (train_run,)


@app.cell
def _(build_solver, mo, train_cfg, train_run):
    with mo.status.spinner(title=f"Training {train_cfg['epochs']} epochs ..."):
        _state, final_metrics, train_seconds = train_run(train_cfg)
    problem, model, solver = build_solver(train_cfg)
    model.load_state_dict(_state)
    model.eval()
    mo.hstack(
        [mo.stat(f"{v:.2e}", label=k) for k, v in final_metrics.items()]
        + [mo.stat(f"{train_seconds:.0f} s", label="training time")],
        wrap=True,
    )
    return final_metrics, problem, solver, train_seconds


@app.cell
def _(LabelTensor, go, mo, np, problem, solver, torch, train_cfg):
    _n = 101
    _xs = np.linspace(0.0, 1.0, _n)
    _X, _Y = np.meshgrid(_xs, _xs)
    _cols = {"x": _X.ravel(), "y": _Y.ravel(), "t": np.full(_X.size, 0.5)}
    _labels = list(problem.input_variables)
    _inp = LabelTensor(
        torch.tensor(np.stack([_cols[v] for v in _labels], 1), dtype=torch.float32),
        _labels,
    )
    with torch.no_grad():
        u_pred = solver.forward(_inp).numpy()[:, 0].reshape(_n, _n)
    _when = " at t = 0.5" if "t" in _labels else ""
    _sol_fig = go.Figure(go.Heatmap(x=_xs, y=_xs, z=u_pred))
    _sol_fig.update_layout(title=f"PINN solution u(x, y){_when}", height=420)
    _panels = [mo.ui.plotly(_sol_fig)]

    poisson_error = None
    if train_cfg["problem"] == "Poisson":
        # -Δu = sin(πx)sin(πy), u = 0 on ∂Ω  →  u = sin(πx)sin(πy) / (2π²)
        _exact = np.sin(np.pi * _X) * np.sin(np.pi * _Y) / (2 * np.pi**2)
        _err = np.abs(u_pred - _exact)
        poisson_error = {
            "rel_l2": float(np.linalg.norm(u_pred - _exact) / np.linalg.norm(_exact)),
            "max_abs": float(_err.max()),
        }
        _err_fig = go.Figure(go.Heatmap(x=_xs, y=_xs, z=_err))
        _err_fig.update_layout(
            title=f"|error| vs exact, relative L2 {poisson_error['rel_l2']:.2%}",
            height=420,
        )
        _panels.append(mo.ui.plotly(_err_fig))
    mo.vstack([mo.md("## Solution"), mo.hstack(_panels, widths="equal")])
    return poisson_error, u_pred


@app.cell
def _(mlflow, mo, pl, train_cfg):
    _heading = mo.md("## Experiment History")
    _experiment_name = f"pina-{train_cfg['problem'].lower().replace(' ', '-')}"
    _runs = mlflow.search_runs(
        experiment_names=[_experiment_name],
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )

    if _runs.empty:
        mo.output.replace(mo.vstack([_heading, mo.md("_No runs yet._")]))
    else:
        _cols = [
            c
            for c in _runs.columns
            if c.startswith(
                ("params.", "metrics.", "tags.mlflow.runName", "run_id", "status")
            )
        ]
        _runs_df = pl.from_pandas(_runs[_cols])
        mo.output.replace(
            mo.vstack([_heading, mo.ui.table(_runs_df, label="Previous Runs")])
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## Optuna Hyperparameter Sweep")
    return


@app.cell
def _(CLI, mo):
    sweep_form = mo.ui.batch(
        mo.md(
            """
            {n_trials} {sweep_epochs} {lr_min} {lr_max}

            {layers_min} {layers_max} {size_min} {size_max}
            """
        ),
        {
            "n_trials": mo.ui.slider(
                2, 50, value=int(CLI.get("trials", 10)), step=1, label="Trials"
            ),
            "sweep_epochs": mo.ui.slider(
                20,
                500,
                value=int(CLI.get("trial-epochs", 200)),
                step=10,
                label="Epochs / trial",
            ),
            "lr_min": mo.ui.number(value=1e-4, label="LR min"),
            "lr_max": mo.ui.number(value=1e-2, label="LR max"),
            "layers_min": mo.ui.slider(1, 4, value=2, step=1, label="Min layers"),
            "layers_max": mo.ui.slider(2, 6, value=4, step=1, label="Max layers"),
            "size_min": mo.ui.slider(16, 64, value=16, step=16, label="Min layer size"),
            "size_max": mo.ui.slider(
                32, 128, value=64, step=16, label="Max layer size"
            ),
        },
    ).form(submit_button_label="Run Optuna Sweep", bordered=True)
    sweep_form
    return (sweep_form,)


@app.cell
def _(
    ACTIVATIONS,
    MLFlowLogger,
    PROBLEM_KINDS,
    ProblemManager,
    SolverManager,
    create_model_for_problem,
    mlflow,
    mo,
    sweep_form,
    torch,
    train_cfg,
    train_solver,
):
    import optuna
    from optuna_integration import PyTorchLightningPruningCallback
    from pina.callback import MetricTracker

    sweep_cfg = sweep_form.value or sweep_form.element.value

    @mo.persistent_cache
    def run_sweep(problem_name: str, cfg: dict) -> optuna.Study:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        exp_name = f"pina-{problem_name.lower().replace(' ', '-')}-sweep"
        mlflow.set_experiment(exp_name)

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("lr", cfg["lr_min"], cfg["lr_max"], log=True)
            n_layers = trial.suggest_int(
                "num_layers", cfg["layers_min"], cfg["layers_max"]
            )
            layer_size = trial.suggest_int(
                "layer_size", cfg["size_min"], cfg["size_max"], step=16
            )
            act = trial.suggest_categorical("activation", list(ACTIVATIONS))
            torch.manual_seed(trial.number)
            problem = ProblemManager.create(PROBLEM_KINDS[problem_name])()
            model = create_model_for_problem(
                problem, layers=[layer_size] * n_layers, activation=ACTIVATIONS[act]
            )
            solver = SolverManager.create(
                "pinn", problem=problem, model=model, learning_rate=lr
            )
            tracker = MetricTracker()
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as run:
                mlflow.log_params(trial.params)
                train_solver(
                    solver,
                    max_epochs=cfg["sweep_epochs"],
                    logger=MLFlowLogger(
                        experiment_name=exp_name,
                        tracking_uri=mlflow.get_tracking_uri(),
                        run_id=run.info.run_id,
                    ),
                    callbacks=[
                        tracker,
                        PyTorchLightningPruningCallback(trial, monitor="train_loss"),
                    ],
                )
                metrics = tracker.metrics
                loss = (
                    float(next(iter(metrics.values())).min())
                    if metrics
                    else float("inf")
                )
                mlflow.log_metric("objective_loss", loss)
            return loss

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=0)
        )
        study.optimize(objective, n_trials=cfg["n_trials"])
        return study

    with mo.status.spinner(title=f"Optuna sweep, {sweep_cfg['n_trials']} trials ..."):
        sweep_study = run_sweep(train_cfg["problem"], sweep_cfg)
    return (sweep_study,)


@app.cell
def _(
    build_optuna_history_figure,
    build_optuna_parallel_figure,
    build_optuna_param_importance_figure,
    build_trials_scatter_chart,
    mo,
    study_trials_dataframe,
    sweep_study,
):
    _df = study_trials_dataframe(sweep_study)
    _best = sweep_study.best_trial

    _best_tab = mo.vstack(
        [
            mo.md(f"**Trial #{_best.number}** — loss `{_best.value:.4e}`"),
            mo.hstack(
                [mo.stat(str(v), label=k) for k, v in _best.params.items()],
                wrap=True,
            ),
        ]
    )

    mo.ui.tabs(
        {
            "Best": _best_tab,
            "Loss Chart": build_trials_scatter_chart(_df, color_by="activation"),
            "Optuna History": build_optuna_history_figure(sweep_study),
            "Param Importance": build_optuna_param_importance_figure(sweep_study),
            "Parallel Coordinates": build_optuna_parallel_figure(sweep_study),
            "All Trials": mo.ui.table(_df, label="Trials"),
        }
    )
    return


if __name__ == "__main__":
    app.run()
