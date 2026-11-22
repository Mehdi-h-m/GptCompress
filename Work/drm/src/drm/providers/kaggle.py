"""Kaggle assisted compute adapter (spec Section 5, 'Kaggle' profile).

Sprint 5 integrates Kaggle as *assisted* execution: the coordinator prepares a
self-contained notebook task bundle and exact launch instructions, the user
runs it on Kaggle, and DRM imports only the declared result files. No Kaggle
SDK calls happen in this sprint — probe records the documented limits as
advertised values, and cancel stays a manual step. API-driven submission and
status arrive in a later sprint once proven against real accounts.

Credentials never enter the bundle: notebook checkpoint pushes resolve the
user's own HF_TOKEN from Kaggle secrets (or the notebook environment).
"""

from __future__ import annotations

from pathlib import Path

from drm.providers.base import ComputeAdapter
from drm.store.db import utcnow

ADVERTISED_LIMITS = {
    "gpu_types": ["P100", "T4 x2"],
    "session_duration_limit_hours": 9,
    "weekly_gpu_quota_hours": 30,
    "output_persistence": "notebook outputs are downloadable; scratch is lost when the session ends",
    "inbound_connectivity": False,
    "parallel_sessions": "only when the verified account policy permits",
    "source": "https://www.kaggle.com/docs/notebooks",
    "documented_at": "2026-11-22",
}


def _kaggle_configured() -> bool:
    """Check for a local kaggle credential without importing the SDK."""
    kaggle_dir = Path.home() / ".kaggle"
    return (kaggle_dir / "kaggle.json").exists() or (kaggle_dir / "kaggle.yaml").exists()


class KaggleCompute(ComputeAdapter):
    """Assisted compute face; deliberately SDK-free for Sprint 5."""

    adapter_name = "kaggle"
    execution_mode = "assisted"

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})

    def discover(self) -> dict:
        return {
            "adapter": self.adapter_name,
            "provider": self.config.get("name", "kaggle"),
            "execution": self.execution_mode,
            "advertised": dict(ADVERTISED_LIMITS),
            "observed": {},
        }

    def probe(self) -> dict:
        return {
            "adapter": self.adapter_name,
            "provider": self.config.get("name", "kaggle"),
            "execution": self.execution_mode,
            "advertised": dict(ADVERTISED_LIMITS),
            "observed": {
                "local_sdk_credentials": _kaggle_configured(),
                "note": "assisted profile: limits come from Kaggle docs, not a live API call",
            },
            "observed_at": utcnow(),
        }

    def quote(self, request: dict) -> dict:
        return {
            "adapter": self.adapter_name,
            "provider": self.config.get("name", "kaggle"),
            "currency": "USD",
            "expires_at": None,
            "billing_increment_seconds": None,
            "prices": {"compute_per_hour": 0.0, "storage_gb_month": 0.0, "transfer_gb": 0.0},
            "credits": {"eligible": True, "available": "free GPU quota, hours tracked by Kaggle"},
            "upper_estimate": {"amount": 0.0, "assumptions": ["free tier; bounded by weekly GPU quota"]},
            "unknowns": ["queue_wait", "session_start_delay"],
        }

    def submit(self, task_bundle: dict) -> dict:
        return {
            "attempt_id": None,
            "manual": True,
            "instructions": (
                "assisted profile: the coordinator emits the bundle and notebook steps;"
                " run them on kaggle.com, then import outputs with drm run import"
            ),
        }

    def status(self, attempt_id: str) -> dict:
        return {"attempt_id": attempt_id, "state": "awaiting_user", "source": "local coordinator record"}

    def cancel(self, attempt_id: str) -> dict:
        return {
            "attempt_id": attempt_id,
            "cancelled": True,
            "confirmed": False,
            "manual_step": "stop the notebook session at kaggle.com; scratch is discarded",
        }

    def list_usage(self) -> dict:
        return {
            "adapter": self.adapter_name,
            "provider": self.config.get("name", "kaggle"),
            "entries": [],
            "unverified_manual_entries": [],
            "settlement_delay": None,
            "quota_note": "weekly GPU quota is visible in Kaggle account settings",
        }

    def notebook_instructions(self, bundle: dict) -> str:
        """Exact steps the user performs on Kaggle for one attempt."""
        return "\n".join(
            [
                "Kaggle assisted run — steps",
                f"  1. New notebook at kaggle.com (GPU ok; internet on only if checkpointing).",
                f"  2. Upload notebook.ipynb from {bundle.get('bundle_dir')} (File > Upload),",
                f"     plus the bundle's inputs/ folder if it has one. Archive: {bundle.get('bundle_zip')}",
                "  3. If the task checkpoints to HF: Add-ons > Secrets > attach HF_TOKEN (write scope).",
                "  4. Run All. The final cell prints the output manifest.",
                "  5. Download the working-dir output (result_manifest.json + outputs/).",
                "  6. Import it here: drm run import " + str(bundle.get("attempt_id")) + " --output <downloaded folder>",
                "  The accepted outputs are only the files declared in result_manifest.json.",
            ]
        )
