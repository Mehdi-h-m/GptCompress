"""drm command-line interface (spec Section 4).

Sprint 0 covers process control and status; catalog and job verbs arrive
with their sprints. Every command accepts ``--json`` and fails with a
stable exit code plus a JSON error record.
"""

from __future__ import annotations

import urllib.error

import typer

import drm
from drm.cli.client import get_json
from drm.cli.exit_codes import ExitCode
from drm.cli.output import print_error, print_payload
from drm.config import base_url, db_path, ensure_state_dir, log_file, pid_file, state_dir
from drm.contracts import ErrorCodes
from drm.daemon import control

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Distributed Resource Manager control CLI",
)
daemon_app = typer.Typer(no_args_is_help=True, help="Start or stop the local control daemon")
app.add_typer(daemon_app, name="daemon")


class CliState:
    json_mode: bool = False
    base: str = ""


def _fail(
    state: CliState,
    code: str,
    message: str,
    exit_code: int = ExitCode.UNEXPECTED_ERROR,
    **details,
) -> None:
    print_error(state.json_mode, code, message, **details)
    raise typer.Exit(exit_code)


@app.callback()
def _root(
    ctx: typer.Context,
    json_mode: bool = typer.Option(False, "--json", help="Emit the versioned JSON envelope"),
) -> None:
    """Control the Distributed Resource Manager daemon and resources."""
    state = CliState()
    state.json_mode = json_mode
    state.base = base_url()
    ctx.obj = state


def _render_status_human(payload: dict) -> str:
    scheduler = payload.get("scheduler", {})
    counts = payload.get("counts", {})
    records = "  ".join(f"{key} {value}" for key, value in counts.items())
    return "\n".join(
        [
            "DRM daemon",
            f"  url        {payload.get('url', '')}",
            f"  status     {payload.get('status', '')}",
            f"  version    {payload.get('version', '')}",
            f"  contract   {payload.get('contract', '')}",
            f"  database   {payload.get('db_path', '')}",
            "  scheduler  "
            f"{scheduler.get('state', '?')} ({scheduler.get('ticks', 0)} ticks,"
            f" last {scheduler.get('last_tick_at') or 'never'})",
            f"  records    {records}",
        ]
    )


_JSON_OPTION = typer.Option(False, "--json", help="Emit the versioned JSON envelope")


@app.command()
def version(ctx: typer.Context, json_mode: bool = _JSON_OPTION) -> None:
    """Print the drm client version."""
    state: CliState = ctx.obj
    state.json_mode = state.json_mode or json_mode
    print_payload(state.json_mode, {"version": drm.__version__}, human=f"drm {drm.__version__}")


@app.command()
def status(ctx: typer.Context, json_mode: bool = _JSON_OPTION) -> None:
    """Show daemon health, scheduler state and record counts."""
    state: CliState = ctx.obj
    state.json_mode = state.json_mode or json_mode
    try:
        payload = get_json(f"{state.base}/status")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _fail(
            state,
            ErrorCodes.DAEMON_UNREACHABLE,
            f"DRM daemon is not reachable at {state.base}",
            url=state.base,
            reason=str(exc),
        )
        return
    except ValueError as exc:
        _fail(
            state,
            ErrorCodes.UNEXPECTED_ERROR,
            f"daemon at {state.base} returned a non-JSON response",
            url=state.base,
            reason=str(exc),
        )
        return
    payload = dict(payload)
    payload["url"] = state.base
    print_payload(state.json_mode, payload, human=_render_status_human(payload))


@daemon_app.command("start")
def daemon_start(
    ctx: typer.Context,
    json_mode: bool = _JSON_OPTION,
    foreground: bool = typer.Option(
        False, "--foreground", help="Run the daemon in this console instead of detaching"
    ),
) -> None:
    """Start the local control daemon (idempotent)."""
    state: CliState = ctx.obj
    state.json_mode = state.json_mode or json_mode
    if control.is_healthy(state.base):
        print_payload(
            state.json_mode,
            {"status": "running", "already_running": True, "url": state.base},
            human=f"DRM daemon already running at {state.base}",
        )
        return
    if foreground:
        import uvicorn

        from drm.config import host, port
        from drm.daemon.app import create_app

        uvicorn.run(create_app(db_path()), host=host(), port=port())
        return
    ensure_state_dir()
    stale = pid_file()
    if stale.exists():
        stale.unlink()  # any pid file without a healthy daemon is stale
    try:
        pid = control.start_detached(state_dir(), log_file())
    except OSError as exc:
        _fail(state, ErrorCodes.UNEXPECTED_ERROR, f"failed to launch daemon process: {exc}")
        return
    if not control.wait_healthy(state.base):
        _fail(
            state,
            ErrorCodes.DAEMON_UNREACHABLE,
            f"daemon did not become healthy within {control.START_TIMEOUT:.0f}s",
            log=str(log_file()),
        )
        return
    print_payload(
        state.json_mode,
        {"status": "running", "pid": pid, "url": state.base, "db_path": str(db_path())},
        human=f"DRM daemon running at {state.base} (pid {pid}, db {db_path()})",
    )


@daemon_app.command("stop")
def daemon_stop(ctx: typer.Context, json_mode: bool = _JSON_OPTION) -> None:
    """Stop the local control daemon (idempotent)."""
    state: CliState = ctx.obj
    state.json_mode = state.json_mode or json_mode
    try:
        result = control.stop(pid_file(), state.base)
    except control.DaemonControlError as exc:
        _fail(state, ErrorCodes.UNEXPECTED_ERROR, str(exc))
        return
    human = (
        f"DRM daemon stopped (pid {result['pid']})"
        if result["was_running"]
        else "DRM daemon was not running"
    )
    print_payload(state.json_mode, result, human=human)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
