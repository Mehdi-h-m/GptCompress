"""drm plan — preview placements with recorded decisions (spec Section 4)."""

from __future__ import annotations

import json

import typer

from drm.cli.common import JSON_OPTION, CliState, request, resolve_state
from drm.cli.output import print_payload

plan_app = typer.Typer(no_args_is_help=True, help="Preview placement plans")


def _parse_param_value(value: str):
    """JSON values (lists/objects) parse; everything else stays a string."""
    if value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _manifest_options(func):
    func = typer.Option(None, "--task", help="Registered task type, e.g. hash_inputs@1")(func)
    func = typer.Option(None, "--input", help="Input artifact id (repeatable)")(func)
    func = typer.Option(None, "--param", help="Task parameter k=v (repeatable)")(func)
    func = typer.Option(None, "--store", help="Preferred output store provider id")(func)
    func = typer.Option(None, "--name", help="Job/task name")(func)
    func = typer.Option(None, "--experiment", help="Experiment id")(func)
    func = typer.Option(None, "--max-attempts", help="Max attempts (default 3)")(func)
    func = typer.Option(None, "--timeout", help="Task timeout seconds (default 300)")(func)
    return func


def _build_payload(task, inputs, params, store, name, experiment, max_attempts, timeout) -> dict:
    payload = {}
    if task is not None:
        payload["task_type"] = task
    if inputs:
        payload["inputs"] = list(inputs)
    if params:
        parsed = {}
        for item in params:
            if "=" not in item:
                raise typer.BadParameter(f"--param expects k=v, got {item!r}")
            key, value = item.split("=", 1)
            parsed[key] = _parse_param_value(value)
        payload["params"] = parsed
    if store is not None:
        payload["output_store"] = store
    if name is not None:
        payload["name"] = name
    if experiment is not None:
        payload["experiment_id"] = experiment
    if max_attempts is not None:
        payload["max_attempts"] = max_attempts
    if timeout is not None:
        payload["timeout_seconds"] = timeout
    return payload


def _render_plan_human(plan_doc: dict) -> str:
    lines = ["DRM plan"]
    chosen = plan_doc.get("chosen")
    if chosen:
        lines.append(f"  compute       {chosen['compute']}")
        lines.append(f"  output store  {chosen['output_store']}")
    else:
        lines.append("  chosen        none (blockers below)")
    cost = plan_doc.get("estimated_cost_range", {})
    lines.append(f"  cost range    {cost.get('min')} - {cost.get('max')} {cost.get('currency')}")
    duration = plan_doc.get("duration_estimate", {})
    lines.append(f"  duration      {duration.get('certainty')}")
    blockers = plan_doc.get("blockers", [])
    if blockers:
        lines.append("  blockers:")
        lines += [f"    - {b}" for b in blockers]
    if plan_doc.get("decision_id"):
        lines.append(f"  decision      {plan_doc['decision_id']} (drm explain)")
    return "\n".join(lines)


@plan_app.command("preview")
def preview(
    ctx: typer.Context,
    task: str = typer.Option(None, "--task"),
    inputs: list[str] = typer.Option(None, "--input"),
    params: list[str] = typer.Option(None, "--param"),
    store: str = typer.Option(None, "--store"),
    name: str = typer.Option(None, "--name"),
    experiment: str = typer.Option(None, "--experiment"),
    max_attempts: int = typer.Option(None, "--max-attempts"),
    timeout: int = typer.Option(None, "--timeout"),
    json_mode: bool = JSON_OPTION,
) -> None:
    """Preview a job placement plan: candidates, cost range and blockers."""
    state: CliState = resolve_state(ctx, json_mode)
    payload = _build_payload(task, inputs, params, store, name, experiment, max_attempts, timeout)
    doc = request(state, "POST", "/plan/preview", payload)
    print_payload(state.json_mode, doc, human=_render_plan_human(doc["plan"]))
