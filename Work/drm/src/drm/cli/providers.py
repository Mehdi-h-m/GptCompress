"""drm providers — connect, list, probe, pause, resume (spec Section 5)."""

from __future__ import annotations

from pathlib import Path

import typer

from drm.cli.common import (
    DRY_RUN_OPTION,
    JSON_OPTION,
    CliState,
    request,
    resolve_state,
)
from drm.cli.output import print_payload

providers_app = typer.Typer(no_args_is_help=True, help="Connect and manage provider instances")


def _provider_line(provider: dict) -> str:
    region = provider.get("region") or "-"
    return (
        f"{provider['id']}  {provider['adapter']:<13} {provider['display_name']:<24}"
        f" {provider['kind']:<8} {provider['state']:<10} {region}"
    )


@providers_app.command("connect")
def connect(
    ctx: typer.Context,
    adapter: str = typer.Argument(..., help="Adapter type: local-runner, filesystem, mock"),
    name: str = typer.Option(..., "--name", help="Display name for this instance"),
    root: Path = typer.Option(None, "--root", help="Store root directory (filesystem adapter)"),
    region: str = typer.Option(None, "--region", help="Region where applicable"),
    credential_ref: str = typer.Option(None, "--credential-ref", help="Credential store reference ID"),
    json_mode: bool = JSON_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Register a provider instance (idempotent per adapter + name)."""
    state: CliState = resolve_state(ctx, json_mode)
    config = {}
    if root is not None:
        config["root"] = str(root)
    payload = {
        "adapter": adapter,
        "name": name,
        "config": config,
        "region": region,
        "credential_ref": credential_ref,
        "dry_run": dry_run,
    }
    doc = request(state, "POST", "/providers", payload)
    if doc.get("dry_run"):
        human = f"would connect {doc['plan']['adapter']} '{doc['plan']['name']}'"
    elif doc.get("already_connected"):
        human = f"provider already connected: {_provider_line(doc['provider'])}"
    else:
        human = f"connected: {_provider_line(doc['provider'])}"
    print_payload(state.json_mode, doc, human=human)


@providers_app.command("list")
def list_providers(ctx: typer.Context, json_mode: bool = JSON_OPTION) -> None:
    """List configured provider instances."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(state, "GET", "/providers")
    providers = doc.get("providers", [])
    lines = ["id  adapter        display name             kind     state      region"]
    lines += [_provider_line(provider) for provider in providers] or ["(none registered)"]
    print_payload(state.json_mode, doc, human="\n".join(lines))


@providers_app.command("probe")
def probe(ctx: typer.Context, provider_id: str = typer.Argument(...), json_mode: bool = JSON_OPTION) -> None:
    """Run live capability checks on one provider instance."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(state, "POST", f"/providers/{provider_id}/probe", {})
    print_payload(state.json_mode, doc, human=f"probe of {provider_id} complete")


@providers_app.command("pause")
def pause(
    ctx: typer.Context,
    provider_id: str = typer.Argument(...),
    json_mode: bool = JSON_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Pause a provider instance; the scheduler stops placing new work there."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(state, "POST", f"/providers/{provider_id}/pause", {"dry_run": dry_run})
    if doc.get("dry_run"):
        human = f"would pause {doc['plan']['provider_id']}"
    else:
        human = f"paused: {_provider_line(doc['provider'])}"
    print_payload(state.json_mode, doc, human=human)


@providers_app.command("resume")
def resume(
    ctx: typer.Context,
    provider_id: str = typer.Argument(...),
    json_mode: bool = JSON_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Resume a paused provider instance."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(state, "POST", f"/providers/{provider_id}/resume", {"dry_run": dry_run})
    if doc.get("dry_run"):
        human = f"would resume {doc['plan']['provider_id']}"
    else:
        human = f"resumed: {_provider_line(doc['provider'])}"
    print_payload(state.json_mode, doc, human=human)
