"""marimo dashboard over the DuckDB provenance store (SPEC §10 / §17.7).

Run with:
    marimo edit examples/02_provenance_dashboard.py

Surfaces task state, agent decisions, experiment outcomes, and
validation verdicts side-by-side so a human can approve/reject runs
without touching MLflow directly.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _header():
    import marimo as mo

    mo.md(
        "# PINA agents — provenance dashboard\n"
        "Reads the DuckDB store written by the multi-agent team. Point "
        "`provenance_db_path` at your local file (default: `provenance.duckdb`)."
    )
    return (mo,)


@app.cell
def _db_picker(mo):
    db_path = mo.ui.text(
        value="provenance.duckdb",
        label="Provenance DuckDB path",
        full_width=True,
    )
    db_path
    return (db_path,)


@app.cell
def _open_store(db_path):
    from marimo_flow.agents.services.provenance import ProvenanceStore

    store = ProvenanceStore(db_path.value)
    return (store,)


@app.cell
def _tasks(mo, store):
    tasks_rows = store.query(
        "SELECT task_id, title, problem_kind, equation_family, "
        "review_required, created_at FROM tasks ORDER BY created_at DESC"
    )
    mo.md("## Tasks")
    mo.ui.table(tasks_rows) if tasks_rows else mo.md("_No tasks recorded yet._")
    return (tasks_rows,)


@app.cell
def _experiments(mo, store):
    experiments_rows = store.query(
        "SELECT experiment_id, task_id, run_id, status, created_at, "
        "finished_at FROM experiments ORDER BY created_at DESC"
    )
    mo.md("## Experiments")
    (
        mo.ui.table(experiments_rows)
        if experiments_rows
        else mo.md("_No experiments recorded yet._")
    )
    return (experiments_rows,)


@app.cell
def _decisions(mo, store):
    decisions_rows = store.query(
        "SELECT created_at, agent, tool, summary, task_id "
        "FROM agent_decisions ORDER BY created_at DESC LIMIT 100"
    )
    mo.md("## Recent agent decisions")
    (
        mo.ui.table(decisions_rows)
        if decisions_rows
        else mo.md("_No decisions recorded yet._")
    )
    return (decisions_rows,)


@app.cell
def _validation(mo, store):
    validation_rows = store.query(
        "SELECT created_at, task_id, run_id, verdict, rationale "
        "FROM validation_reports ORDER BY created_at DESC LIMIT 50"
    )
    mo.md("## Validation verdicts")
    (
        mo.ui.table(validation_rows)
        if validation_rows
        else mo.md("_No validation reports yet._")
    )
    return (validation_rows,)


@app.cell
def _handoffs(mo, store):
    handoffs_rows = store.query(
        "SELECT created_at, from_agent, to_agent, reason, task_id "
        "FROM handoff_records ORDER BY created_at DESC LIMIT 50"
    )
    mo.md("## Handoffs")
    (
        mo.ui.table(handoffs_rows)
        if handoffs_rows
        else mo.md("_No handoffs recorded yet._")
    )
    return (handoffs_rows,)


@app.cell
def _preset_family_picker(mo):
    family = mo.ui.dropdown(
        options=["problem", "model", "solver"],
        value="problem",
        label="Preset family",
    )
    include_deprecated = mo.ui.checkbox(value=False, label="include deprecated")
    mo.md("## Preset-Bibliothek")
    mo.hstack([family, include_deprecated])
    return (family, include_deprecated)


@app.cell
def _presets(mo, store, family, include_deprecated):
    preset_table_map = {
        "problem": "preset_problems",
        "model": "preset_models",
        "solver": "preset_solvers",
    }
    preset_table = preset_table_map[family.value]
    preset_where = "" if include_deprecated.value else "WHERE status = 'active'"
    presets_rows = store.query(
        f"SELECT name, description, status, author, tags, parent_name, "
        f"       created_at "
        f"FROM {preset_table} {preset_where} "
        f"ORDER BY created_at DESC"
    )
    if presets_rows:
        mo.ui.table(presets_rows)
    else:
        mo.md(
            f"_No {family.value} compositions yet — they get authored "
            "by the agents as they go._"
        )
    return (presets_rows,)


@app.cell
def _preset_problem_detail_picker(mo, store):
    preset_name_rows = store.query(
        "SELECT name FROM preset_problems WHERE status = 'active' "
        "ORDER BY created_at DESC LIMIT 50"
    )
    preset_names = [r["name"] for r in preset_name_rows]
    picker = mo.ui.dropdown(
        options=preset_names or ["—"],
        value=preset_names[0] if preset_names else "—",
        label="Inspect problem preset (3D view)",
    )
    picker
    return (picker,)


@app.cell
def _problem_3d_view(mo, store, picker):
    # Single return point — marimo's static analyser flags `return` inside
    # an `if` branch in cell bodies as "return outside function".
    if picker.value == "—":
        view_md = mo.md("_No stored problem compositions to visualise yet._")
    else:
        payload_rows = store.query(
            "SELECT payload FROM preset_problems WHERE name = ? AND status = 'active'",
            [picker.value],
        )
        if not payload_rows:
            view_md = mo.md("_Preset disappeared — refresh._")
        else:
            from marimo_flow.agents.schemas import ProblemSpec
            from marimo_flow.agents.services.composer import compose_problem
            from marimo_flow.core.viz3d import domain_figure

            spec_dict = payload_rows[0]["payload"]
            if isinstance(spec_dict, str):
                import json

                spec_dict = json.loads(spec_dict)
            spec = ProblemSpec.model_validate(spec_dict)
            try:
                problem = compose_problem(spec)()
                preset_fig = domain_figure(problem)
                view_md = mo.vstack(
                    [
                        mo.md(f"### Spatial domain of `{picker.value}`"),
                        mo.ui.plotly(preset_fig),
                    ]
                )
            except Exception as exc:  # noqa: BLE001 — user-facing surface
                view_md = mo.md(f"**Failed to render 3D view**: {exc}")
    view_md
    return (view_md,)


if __name__ == "__main__":
    app.run()
