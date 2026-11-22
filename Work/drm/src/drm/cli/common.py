"""Shared CLI plumbing: state, error-to-exit-code mapping, daemon requests.

Error-code to exit-code mapping (spec Section 4, rule 3): ``validation`` and
``policy_blocked`` exit 2; ``awaiting_user`` exits 3; ``reconcile_required``
exits 4; everything else exits 1.
"""

from __future__ import annotations

import json
import time
import urllib.error
from typing import NoReturn

import typer

from drm.cli.client import get_json, post_json
from drm.cli.exit_codes import ExitCode
from drm.cli.output import print_error
from drm.contracts import ErrorCodes, OPERATION_TERMINAL_STATES

JSON_OPTION = typer.Option(False, "--json", help="Emit the versioned JSON envelope")
DRY_RUN_OPTION = typer.Option(False, "--dry-run", help="Show what would happen without dispatching")
IDEMPOTENCY_OPTION = typer.Option(
    None, "--idempotency-key", help="Key that makes retries return the original operation"
)


class CliState:
    json_mode: bool = False
    base: str = ""


def resolve_state(ctx: typer.Context, json_flag: bool) -> CliState:
    state: CliState = ctx.obj
    state.json_mode = state.json_mode or json_flag
    return state


def exit_code_for(code: str) -> int:
    mapping = {
        "validation": ExitCode.POLICY_BLOCKED,
        "policy_blocked": ExitCode.POLICY_BLOCKED,
        "awaiting_user": ExitCode.AWAITING_USER,
        "reconcile_required": ExitCode.RECONCILE_REQUIRED,
    }
    return mapping.get(code, ExitCode.UNEXPECTED_ERROR)


def fail(state: CliState, code: str, message: str, exit_code: int | None = None, **details) -> NoReturn:
    print_error(state.json_mode, code, message, **details)
    raise typer.Exit(exit_code if exit_code is not None else exit_code_for(code))


def request(state: CliState, method: str, path: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    """Call the daemon; daemon error records become CLI failures with mapped exit codes."""
    url = f"{state.base}{path}"
    try:
        if method == "GET":
            return get_json(url, timeout=timeout)
        return post_json(url, payload or {}, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode() or "{}")
        except (ValueError, OSError):
            body = {}
        error = body.get("error") if isinstance(body, dict) else None
        if error:
            fail(state, error["code"], error.get("message", ""), **error.get("details", {}))
        fail(state, ErrorCodes.UNEXPECTED_ERROR, f"daemon returned HTTP {exc.code} for {path}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        fail(
            state,
            ErrorCodes.DAEMON_UNREACHABLE,
            f"DRM daemon is not reachable at {state.base}",
            url=state.base,
            reason=str(exc),
        )
    except ValueError as exc:
        fail(state, ErrorCodes.UNEXPECTED_ERROR, f"daemon returned a non-JSON response", url=url, reason=str(exc))


def wait_for_operation(state: CliState, op_id: str, timeout: float, poll_interval: float = 0.5) -> dict:
    """Poll one operation to a terminal state; fail with ``timeout`` otherwise."""
    deadline = time.monotonic() + timeout
    while True:
        doc = request(state, "GET", f"/operations/{op_id}")
        operation = doc.get("operation", {})
        if operation.get("state") in {state.value for state in OPERATION_TERMINAL_STATES}:
            return doc
        if time.monotonic() >= deadline:
            fail(
                state,
                ErrorCodes.TIMEOUT,
                f"operation {op_id} did not reach a terminal state within {timeout:g}s",
                operation=op_id,
                state=operation.get("state"),
            )
        time.sleep(poll_interval)
