"""Select measured run pairs and preserve missing values for the desktop charts."""

import math


def measured(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def resource_values(run):
    if not run or run.get("status") != "complete":
        return None, None, None
    resources = run.get("resources", {})
    return (measured(run.get("seconds")), measured(resources.get("peak_rss_bytes")),
            measured(resources.get("peak_vram_bytes")))


def resource_runs(version):
    return [run for run in version.get("experiments", []) if any(v is not None for v in resource_values(run))]


def matched_workload(a, b):
    """Check recorded benchmark inputs before computing a resource delta."""
    if not a or not b or a.get("status") != "complete" or b.get("status") != "complete":
        return False
    identity = ("protocol_hash", "device", "cpu_threads")
    if any(a.get(key) is None or a.get(key) != b.get(key) for key in identity):
        return False
    question_ids = a.get("question_ids", {})
    if not question_ids or question_ids != b.get("question_ids"):
        return False
    for run in (a, b):
        tasks = run.get("tasks", {})
        if tasks.keys() != question_ids.keys():
            return False
        for name, ids in question_ids.items():
            task = tasks[name]
            if not ids or task.get("status") != "complete" or task.get("n") != len(ids):
                return False
    return True


def default_runs(left, right):
    return next(((a, b) for a in left for b in right if matched_workload(a, b)),
                (next(iter(left), None), next(iter(right), None)))


def relative_delta(a, b):
    return 100 * (b - a) / a if a and b is not None else None
