# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # Pulls pina-mathlab, torch, plotly, ... Declared explicitly so marimo's
#     # sandbox does not auto-install the unrelated PyPI package "pina".
#     "marimo-flow>=0.4.3",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import mlflow
    import polars as pl
    import torch
    import torch.nn as nn
    from lightning.pytorch.loggers import MLFlowLogger

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

    return (
        MLFlowLogger,
        FoundationModelAdapter,
        ProblemManager,
        SolverManager,
        create_model_for_problem,
        build_optuna_history_figure,
        build_optuna_parallel_figure,
        build_optuna_param_importance_figure,
        build_trials_scatter_chart,
        mlflow,
        mo,
        nn,
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

        Select a PDE, configure the network architecture, train, and inspect results.
        All runs are tracked in MLflow.
        """
    )
    return


@app.cell
def _(mo, nn):
    problem_type = mo.ui.dropdown(
        options=["Poisson", "Heat Equation", "Wave Equation"],
        value="Poisson",
        label="Problem",
    )
    backbone = mo.ui.dropdown(
        options=["baseline", "foundation"],
        value="baseline",
        label="Backbone",
    )
    model_id = mo.ui.text(
        value="polymathic-ai/walrus",
        label="HuggingFace model id",
    )
    solver_type = mo.ui.dropdown(
        options=["PINN", "SAPINN"],
        value="PINN",
        label="Solver",
    )
    layer_size = mo.ui.slider(16, 128, value=64, step=16, label="Layer size")
    num_layers = mo.ui.slider(2, 5, value=3, step=1, label="Layers")
    activation = mo.ui.dropdown(
        options={
            "Tanh": nn.Tanh,
            "SiLU": nn.SiLU,
            "Softplus": nn.Softplus,
        },
        value="Tanh",
        label="Activation",
    )
    lr = mo.ui.number(value=1e-3, label="Learning rate")
    epochs = mo.ui.slider(100, 2000, value=400, step=50, label="Epochs")
    freeze = mo.ui.switch(value=True, label="Freeze backbone")

    mo.hstack(
        [
            mo.vstack([problem_type, backbone, solver_type, model_id]),
            mo.vstack([layer_size, num_layers, activation]),
            mo.vstack([lr, epochs, freeze]),
        ],
        widths="equal",
    )
    return (
        activation,
        backbone,
        epochs,
        model_id,
        freeze,
        layer_size,
        lr,
        num_layers,
        problem_type,
        solver_type,
    )


@app.cell
def _(ProblemManager, problem_type):
    _kind = {"Poisson": "poisson", "Heat Equation": "heat", "Wave Equation": "wave"}[
        problem_type.value
    ]
    problem_class = ProblemManager.create(_kind)
    problem = problem_class()
    is_time_dependent = problem_type.value in ("Heat Equation", "Wave Equation")
    return is_time_dependent, problem


@app.cell
def _(
    FoundationModelAdapter,
    SolverManager,
    create_model_for_problem,
    activation,
    backbone,
    model_id,
    freeze,
    layer_size,
    lr,
    num_layers,
    problem,
    solver_type,
):
    if backbone.value == "foundation":
        model = FoundationModelAdapter(
            checkpoint=model_id.value.strip() or "polymathic-ai/walrus",
            input_dimensions=len(problem.input_variables),
            freeze_backbone=freeze.value,
            out_labels=tuple(problem.output_variables),
        )
    else:
        layers = [layer_size.value] * num_layers.value
        model = create_model_for_problem(
            problem, layers=layers, activation=activation.value
        )

    _solver_kind = "sapinn" if solver_type.value == "SAPINN" else "pinn"
    solver = SolverManager.create(
        _solver_kind, problem=problem, model=model, learning_rate=lr.value
    )
    return model, solver


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start training")
    train_button
    return (train_button,)


@app.cell
def _(
    MLFlowLogger,
    backbone,
    epochs,
    lr,
    mlflow,
    mo,
    problem_type,
    solver,
    solver_type,
    train_button,
    train_solver,
):
    mo.stop(not train_button.value, "Click the button to start training.")

    _experiment_name = f"pina-{problem_type.value.lower().replace(' ', '-')}"
    mlflow.set_experiment(_experiment_name)

    logger = MLFlowLogger(
        experiment_name=_experiment_name,
        tracking_uri=mlflow.get_tracking_uri(),
        log_model=True,
        run_name=f"{backbone.value}-{solver_type.value}-{epochs.value}ep",
    )
    logger.log_hyperparams(
        {
            "problem": problem_type.value,
            "backbone": backbone.value,
            "solver": solver_type.value,
            "learning_rate": lr.value,
            "max_epochs": epochs.value,
        }
    )

    trainer = train_solver(
        solver,
        max_epochs=epochs.value,
        logger=logger,
        callbacks=[],
    )
    final_metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    return final_metrics, logger, trainer


@app.cell
def _(
    final_metrics,
    mo,
    problem_type,
    trainer,
):
    mo.stop(trainer is None, "")

    _equation_tab = mo.vstack(
        [
            mo.md(f"### {problem_type.value}"),
            mo.md(
                {
                    "Poisson": r"$-\nabla^2 u = \sin(\pi x)\sin(\pi y)$",
                    "Heat Equation": r"$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$",
                    "Wave Equation": r"$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$",
                }[problem_type.value]
            ),
            mo.md(
                {
                    "Poisson": "Dirichlet BCs: $u = 0$ on $\\partial\\Omega$, domain $[0,1]^2$",
                    "Heat Equation": "IC: $u(x,y,0) = \\sin(\\pi x)\\sin(\\pi y)$, homogeneous Dirichlet BCs",
                    "Wave Equation": "IC: $u(x,y,0) = \\sin(\\pi x)\\sin(\\pi y)$, $u_t(x,y,0) = 0$",
                }[problem_type.value]
            ),
        ]
    )

    _metric_cards = mo.hstack(
        [mo.stat(f"{v:.4f}", label=k) for k, v in final_metrics.items()],
        wrap=True,
    )

    _results_tab = mo.vstack(
        [
            mo.md("## Results"),
            _metric_cards,
            mo.callout(
                "Visualization is now focused on Optuna study analytics below.",
                kind="info",
            ),
        ]
    )

    _training_tab = mo.vstack(
        [
            mo.md("## Training Metrics"),
            _metric_cards,
        ]
    )

    mo.ui.tabs(
        {
            "Equation": _equation_tab,
            "Training": _training_tab,
            "Results": _results_tab,
        }
    )
    return


@app.cell
def _(mlflow, mo, pl, problem_type):
    mo.md("## Experiment History")

    _experiment_name = f"pina-{problem_type.value.lower().replace(' ', '-')}"
    _runs = mlflow.search_runs(
        experiment_names=[_experiment_name],
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )

    if _runs.empty:
        mo.output.replace(mo.md("_No runs yet._"))
    else:
        _cols = [
            c
            for c in _runs.columns
            if c.startswith(
                ("params.", "metrics.", "tags.mlflow.runName", "run_id", "status")
            )
        ]
        _runs_df = pl.from_pandas(_runs[_cols])
        mo.output.replace(mo.ui.table(_runs_df, label="Previous Runs"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## Optuna Hyperparameter Sweep")
    return


@app.cell
def _(mo):
    n_trials = mo.ui.slider(5, 50, value=10, step=5, label="Trials")
    sweep_epochs = mo.ui.slider(50, 500, value=200, step=50, label="Epochs / trial")
    lr_min = mo.ui.number(value=1e-4, label="LR min")
    lr_max = mo.ui.number(value=1e-2, label="LR max")
    sweep_layers_min = mo.ui.slider(1, 4, value=2, step=1, label="Min layers")
    sweep_layers_max = mo.ui.slider(2, 6, value=4, step=1, label="Max layers")
    sweep_size_min = mo.ui.slider(16, 64, value=16, step=16, label="Min layer size")
    sweep_size_max = mo.ui.slider(32, 128, value=64, step=16, label="Max layer size")

    mo.hstack(
        [
            mo.vstack([n_trials, sweep_epochs]),
            mo.vstack([lr_min, lr_max]),
            mo.vstack([sweep_layers_min, sweep_layers_max]),
            mo.vstack([sweep_size_min, sweep_size_max]),
        ],
        widths="equal",
    )
    return (
        lr_max,
        lr_min,
        n_trials,
        sweep_epochs,
        sweep_layers_max,
        sweep_layers_min,
        sweep_size_max,
        sweep_size_min,
    )


@app.cell
def _(mo):
    sweep_button = mo.ui.run_button(label="Run Optuna Sweep")
    sweep_button
    return (sweep_button,)


@app.cell
def _(
    MLFlowLogger,
    ProblemManager,
    SolverManager,
    create_model_for_problem,
    lr_max,
    lr_min,
    mlflow,
    mo,
    n_trials,
    nn,
    problem_type,
    sweep_button,
    sweep_epochs,
    sweep_layers_max,
    sweep_layers_min,
    sweep_size_max,
    sweep_size_min,
    train_solver,
):
    import optuna
    from optuna_integration import PyTorchLightningPruningCallback
    from pina.callback import MetricTracker

    mo.stop(not sweep_button.value, "Click 'Run Optuna Sweep' to start.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    _act_map = {"Tanh": nn.Tanh, "SiLU": nn.SiLU, "Softplus": nn.Softplus}
    _exp_name = f"pina-{problem_type.value.lower().replace(' ', '-')}-sweep"
    mlflow.set_experiment(_exp_name)

    def _objective(trial: optuna.Trial) -> float:
        _lr = trial.suggest_float("lr", lr_min.value, lr_max.value, log=True)
        _n_layers = trial.suggest_int(
            "num_layers", sweep_layers_min.value, sweep_layers_max.value
        )
        _layer_size = trial.suggest_int(
            "layer_size", sweep_size_min.value, sweep_size_max.value, step=16
        )
        _act = trial.suggest_categorical("activation", ["Tanh", "SiLU", "Softplus"])

        _kind = {
            "Poisson": "poisson",
            "Heat Equation": "heat",
            "Wave Equation": "wave",
        }[problem_type.value]
        _problem_cls = ProblemManager.create(_kind)
        _problem = _problem_cls()
        _model = create_model_for_problem(
            _problem,
            layers=[_layer_size] * _n_layers,
            activation=_act_map[_act],
        )
        _solver = SolverManager.create(
            "pinn", problem=_problem, model=_model, learning_rate=_lr
        )

        _tracker = MetricTracker()
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as _run:
            mlflow.log_params(trial.params)
            _ml_logger = MLFlowLogger(
                experiment_name=_exp_name,
                tracking_uri=mlflow.get_tracking_uri(),
                run_id=_run.info.run_id,
            )
            _trainer = train_solver(
                _solver,
                max_epochs=sweep_epochs.value,
                logger=_ml_logger,
                callbacks=[
                    _tracker,
                    PyTorchLightningPruningCallback(trial, monitor="train_loss"),
                ],
            )
            _metrics = _tracker.metrics
            _loss = (
                float(next(iter(_metrics.values())).min()) if _metrics else float("inf")
            )
            mlflow.log_metric("objective_loss", _loss)

        mo.output.replace(
            mo.md(
                f"Trial **{trial.number + 1} / {n_trials.value}** — loss: `{_loss:.4e}`"
            )
        )
        return _loss

    sweep_study = optuna.create_study(direction="minimize")
    sweep_study.optimize(_objective, n_trials=n_trials.value)
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
    mo.stop(sweep_study is None, "")

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

    _loss_chart = build_trials_scatter_chart(_df, color_by="activation")
    _history_fig = build_optuna_history_figure(sweep_study)
    _importance_fig = build_optuna_param_importance_figure(sweep_study)
    _parallel_fig = build_optuna_parallel_figure(sweep_study)

    mo.ui.tabs(
        {
            "Best": _best_tab,
            "Loss Chart": _loss_chart,
            "Optuna History": _history_fig,
            "Param Importance": _importance_fig,
            "Parallel Coordinates": _parallel_fig,
            "All Trials": mo.ui.table(_df, label="Trials"),
        }
    )
    return


if __name__ == "__main__":
    app.run()
