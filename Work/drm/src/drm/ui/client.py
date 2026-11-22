"""Desktop UI client of the loopback daemon.

The GUI is a client of the same daemon the CLI uses (spec Section 3) — no
direct database access, so the GUI and the CLI cannot drift apart.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class DaemonError(Exception):
    """A daemon error record, or a connection failure."""

    def __init__(self, code: str, message: str, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class UiClient:
    def __init__(self, base_url: str | None = None) -> None:
        if base_url is None:
            from drm.config import base_url as config_base_url

            base_url = config_base_url()
        self.base_url = base_url.rstrip("/")

    # --- transport -----------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
        url = f"{self.base_url}{path}"
        try:
            if method == "GET":
                request = urllib.request.Request(url)
            else:
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload or {}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method=method,
                )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read().decode() or "{}")
            except (ValueError, OSError):
                body = {}
            error = body.get("error") if isinstance(body, dict) else None
            if error:
                raise DaemonError(error["code"], error.get("message", ""), status=exc.code)
            raise DaemonError("unexpected_error", f"daemon returned HTTP {exc.code}", status=exc.code)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DaemonError("daemon_unreachable", f"daemon is not reachable at {self.base_url}: {exc}")
        except ValueError as exc:
            raise DaemonError("unexpected_error", f"daemon returned a non-JSON response: {exc}")

    def get(self, path: str, timeout: float = 10.0) -> dict:
        return self._request("GET", path, timeout=timeout)

    def post(self, path: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
        return self._request("POST", path, payload, timeout=timeout)

    # --- daemon ----------------------------------------------------------------

    def health(self) -> dict:
        return self.get("/health", timeout=3.0)

    def status(self) -> dict:
        return self.get("/status")

    # --- providers ---------------------------------------------------------------

    def providers(self) -> list[dict]:
        return self.get("/providers").get("providers", [])

    def connect_provider(self, payload: dict) -> dict:
        return self.post("/providers", payload)

    def probe_provider(self, provider_id: str) -> dict:
        return self.post(f"/providers/{provider_id}/probe", {})

    def set_provider_state(self, provider_id: str, action: str) -> dict:
        return self.post(f"/providers/{provider_id}/{action}", {})

    # --- data catalog ------------------------------------------------------------

    def artifacts(self) -> list[dict]:
        return self.get("/data/artifacts").get("artifacts", [])

    def artifact(self, artifact_id: str) -> dict:
        return self.get(f"/data/artifacts/{artifact_id}").get("artifact", {})

    def ingest(self, payload: dict) -> dict:
        return self.post("/data/ingest", payload, timeout=30.0)

    def pin(self, artifact_id: str) -> dict:
        return self.post("/data/pin", {"artifact_id": artifact_id})

    def evict(self, copy_id: str) -> dict:
        return self.post("/data/evict", {"copy_id": copy_id})

    def verify(self, copy_id: str) -> dict:
        return self.post("/data/verify", {"copy_id": copy_id})

    # --- plan and runs -------------------------------------------------------------

    def preview_plan(self, payload: dict) -> dict:
        return self.post("/plan/preview", payload, timeout=30.0)

    def submit_run(self, payload: dict) -> dict:
        return self.post("/run/submit", payload, timeout=30.0)

    def run(self, job_id: str) -> dict:
        return self.get(f"/run/{job_id}")

    def jobs(self) -> list[dict]:
        return self.get("/jobs").get("jobs", [])

    def cancel_run(self, job_id: str) -> dict:
        return self.post(f"/run/{job_id}/cancel", {})

    def retry_job(self, job_id: str) -> dict:
        return self.post(f"/jobs/{job_id}/retry", {})

    # --- operations and decisions ---------------------------------------------------

    def operation(self, op_id: str) -> dict:
        return self.get(f"/operations/{op_id}")

    def decision(self, decision_id: str) -> dict:
        return self.get(f"/decisions/{decision_id}")
