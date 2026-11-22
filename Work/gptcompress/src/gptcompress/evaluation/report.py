"""Render local experiment records without third-party dependencies."""

from __future__ import annotations

import html
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def _escape(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return html.escape(str(value))


def _number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _format(value: Any, divisor: float = 1, suffix: str = "") -> str:
    return f"{value / divisor:,.2f}{suffix}" if _number(value) else "Not measured"


def _score(task: dict) -> str:
    value = task.get("score")
    if not _number(value):
        return "Not measured"
    result = f"{100 * value:.2f}%"
    error = task.get("stderr")
    if _number(error):
        result += f" (SE {100 * error:.2f} pp)"
    return result


def _read_json(path: Path, warnings: list[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        warnings.append(f"{path.name}: {error}")
        return {}


def _read_runs(root: Path) -> list[dict]:
    runs = []
    for directory in sorted((root / "experiments").glob("*")):
        if not directory.is_dir():
            continue
        if (directory / "progress.json").exists() and not (directory / "manifest.json").exists():
            continue
        warnings: list[str] = []
        manifest = _read_json(directory / "manifest.json", warnings)
        metrics = _read_json(directory / "metrics.json", warnings)
        predictions = []
        prediction_path = directory / "predictions.jsonl"
        if prediction_path.exists():
            with prediction_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("Expected a JSON object")
                        predictions.append(record)
                    except ValueError as error:
                        warnings.append(f"predictions.jsonl line {line_number}: {error}")
        runs.append({"id": manifest.get("run_id", directory.name), "manifest": manifest,
                     "metrics": metrics, "predictions": predictions, "warnings": warnings})
    return runs


def _tasks(run: dict) -> dict:
    tasks = run["metrics"].get("tasks", {})
    return {name: value for name, value in tasks.items() if isinstance(value, dict)} if isinstance(tasks, dict) else {}


def _delta(run: dict, task_name: str, reference: dict | None) -> str:
    if reference is None:
        return "Reference unavailable"
    protocol = run["manifest"].get("protocol_hash")
    if not protocol or protocol != reference["manifest"].get("protocol_hash"):
        return "Different or unknown protocol"
    current = _tasks(run).get(task_name, {})
    previous = _tasks(reference).get(task_name, {})
    if current.get("n") != previous.get("n"):
        return "Different sample counts"
    if current.get("status") not in ("complete", "completed") or previous.get("status") not in ("complete", "completed"):
        return "Incomplete task"
    if not _number(current.get("score")) or not _number(previous.get("score")):
        return "Not measured"
    return f"{100 * (current['score'] - previous['score']):+.2f} pp"


def _summary(runs: list[dict]) -> str:
    by_id = {run["id"]: run for run in runs}
    rows = []
    for index, run in enumerate(runs):
        manifest = run["manifest"]
        for task_name, task in _tasks(run).items():
            baseline = by_id.get(manifest.get("baseline_run_id"))
            parent = by_id.get(manifest.get("parent_run_id"))
            fields = [task_name, _score(task), task.get("n", "Unknown"), task.get("status", "Unknown"),
                      _delta(run, task_name, baseline), _delta(run, task_name, parent)]
            cells = "".join(f"<td>{_escape(value)}</td>" for value in fields)
            rows.append(f'<tr><td><a href="#run-{index}">{_escape(run["id"])}</a></td>{cells}</tr>')
    if not rows:
        return "<p>No benchmark measurements are available yet.</p>"
    headers = ["Run", "Benchmark", "Score", "Samples", "Status", "Change from baseline", "Change from parent"]
    return '<div class="scroll"><table><thead><tr>' + "".join(f"<th>{heading}</th>" for heading in headers) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _plots(runs: list[dict]) -> str:
    groups: dict[tuple, list] = {}
    for run in runs:
        manifest = run["manifest"]
        size = manifest.get("model_size_bytes")
        protocol = manifest.get("protocol_hash")
        if not protocol or not _number(size) or size <= 0:
            continue
        for name, task in _tasks(run).items():
            if _number(task.get("score")) and task.get("status") in ("complete", "completed"):
                groups.setdefault((protocol, name, task.get("n")), []).append((size / 1e9, task["score"] * 100, run["id"]))
    plots = []
    for (protocol, task, count), points in groups.items():
        if len(points) < 2:
            continue
        max_size = max(point[0] for point in points) * 1.1
        circles = []
        legend = []
        for size, score, run_id in points:
            x, y = 60 + size / max_size * 570, 260 - score / 100 * 220
            label = f"{run_id}: {size:.3f} GB, {score:.2f}%"
            circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5"><title>{_escape(label)}</title></circle>')
            legend.append(f"<li>{_escape(label)}</li>")
        plots.append(f'<figure><figcaption>{_escape(task)} — {_escape(count)} examples, protocol {_escape(str(protocol)[:12])}</figcaption>'
                     '<svg viewBox="0 0 700 310" role="img" aria-label="Benchmark score versus checkpoint size">'
                     '<path d="M60 40V260H630" fill="none" stroke="currentColor"/>'
                     '<text x="20" y="45">100%</text><text x="25" y="264">0%</text>'
                     f'<text x="60" y="285">0 GB</text><text x="580" y="285">{max_size:.2f} GB</text>'
                     '<text x="250" y="305">Checkpoint size (decimal GB)</text>' + "".join(circles) + '</svg><ul>' + "".join(legend) + '</ul></figure>')
    return "".join(plots)


def _run_detail(run: dict, index: int) -> str:
    manifest, metrics = run["manifest"], run["metrics"]
    resources = metrics.get("resources", {}) or {}
    facts = {
        "Model": manifest.get("model_id", "Unknown"),
        "Revision": manifest.get("model_revision", "Unknown"),
        "Profile": manifest.get("profile", "Unknown"),
        "Created": manifest.get("created_at", "Unknown"),
        "Status": metrics.get("status", manifest.get("status", "Unknown")),
        "Checkpoint size": _format(manifest.get("model_size_bytes"), 1e9, " GB"),
        "Complete asset size": _format(manifest.get("asset_size_bytes"), 1e9, " GB"),
        "Peak RAM (RSS)": _format(resources.get("peak_rss_bytes"), 2**30, " GiB"),
        "Peak VRAM": _format(resources.get("peak_vram_bytes"), 2**30, " GiB"),
        "Elapsed time": _format(resources.get("elapsed_seconds"), suffix=" seconds"),
        "Model load time": _format(metrics.get("model_load_seconds"), suffix=" seconds"),
        "Error": metrics.get("error") or "None recorded",
    }
    details = "".join(f"<dt>{label}</dt><dd>{_escape(value)}</dd>" for label, value in facts.items())
    warnings = "".join(f'<p class="warning">{_escape(message)}</p>' for message in run["warnings"])
    warnings += "".join(
        f'<p class="warning">{_escape(name)} &middot; {_escape(task["error"])}</p>'
        for name, task in _tasks(run).items() if task.get("error")
    )
    answers = []
    for prediction in run["predictions"]:
        label = f"{prediction.get('task', 'Unknown task')} / {prediction.get('doc_id', 'Unknown ID')}"
        notes = []
        if prediction.get("input_truncated"):
            notes.append("Input exceeded the context budget")
        if prediction.get("output_near_cap"):
            notes.append("Response near output limit based on re-tokenization")
        fields = "".join(f"<h4>{label}</h4><pre>{_escape(prediction.get(key, 'Unavailable'))}</pre>"
                         for key, label in [("prompt", "Prompt"), ("response", "Model response"), ("target", "Reference answer"), ("score", "Score")])
        if notes:
            fields = '<p class="warning">' + _escape(". ".join(notes)) + "</p>" + fields
        answers.append(f"<details><summary>{_escape(label)}</summary>{fields}</details>")
    if not answers:
        answers.append("<p>No per-example predictions are available.</p>")
    reconstruction = manifest.get("reconstruction", [])
    if reconstruction:
        rows = []
        for matrix in reconstruction:
            values = [matrix.get("name"), matrix.get("rank", "Skipped"),
                      matrix.get("original_parameters"), matrix.get("factor_parameters"),
                      matrix.get("stored_relative_squared_error"), matrix.get("iterations")]
            rows.append("<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in values) + "</tr>")
        headers = ["Matrix", "Rank", "Original parameters", "Factor parameters",
                   "Stored relative squared error", "ALS iterations"]
        table = "<table><thead><tr>" + "".join(f"<th>{label}</th>" for label in headers)
        table += "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        answers.insert(0, '<details><summary>Weight reconstruction</summary><div class="scroll">'
                       + table + "</div></details>")
    return f'<section id="run-{index}"><h2>{_escape(run["id"])}</h2>{warnings}<dl>{details}</dl>' + "".join(answers) + f'<details><summary>Complete manifest</summary><pre>{_escape(manifest)}</pre></details></section>'


def _asset_status(root: Path) -> str:
    items = []
    for path in sorted((root / "datasets" / "benchmarks").glob("*/manifest.json")):
        warnings = []
        asset = _read_json(path, warnings)
        fields = (
            path.parent.name, asset.get("status", "Unknown"), asset.get("revision") or "Unavailable",
            asset.get("license") or "Unspecified by source", asset.get("error") or " ".join(warnings),
        )
        items.append("<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in fields) + "</tr>")
    if not items:
        return ""
    headings = ("Benchmark", "Data status", "Revision", "License", "Access details")
    headers = "".join(f"<th>{label}</th>" for label in headings)
    return '<section><h2>Benchmark data</h2><div class="scroll"><table><thead><tr>' + headers + (
        "</tr></thead><tbody>" + "".join(items) + "</tbody></table></div></section>"
    )


def generate_report(root: Path) -> Path:
    """Write an atomic standalone report, leaving experiment artifacts untouched."""
    root = Path(root)
    runs = _read_runs(root)
    style = """
body{font-family:system-ui,sans-serif;margin:2rem auto;padding:0 1.5rem;max-width:1300px;color:#172033;background:#f5f7fb}
h1,h2{line-height:1.2}a{color:#1555a0}section,figure{background:white;border:1px solid #d6dce6;padding:1.25rem;margin:1.5rem 0;border-radius:8px}
table{border-collapse:collapse;width:100%;background:white}th,td{padding:.65rem;text-align:left;border-bottom:1px solid #d6dce6;vertical-align:top}th{background:#e9eef5}
.scroll{overflow-x:auto}dl{display:grid;grid-template-columns:12rem 1fr;gap:.4rem}dt{font-weight:600}dd{margin:0;overflow-wrap:anywhere}
details{border-top:1px solid #d6dce6;padding:.7rem 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fb;padding:.8rem}
svg{max-width:700px;width:100%;display:block}circle{fill:#176e8c}svg text{font-size:12px;fill:#172033}.warning{color:#8a3c00}figcaption{font-weight:600}
@media(max-width:600px){dl{grid-template-columns:1fr}dd{margin-bottom:.5rem}}
"""
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += f'<title>GptCompress experiments</title><style>{style}</style></head><body><h1>GptCompress experiments</h1>'
    document += '<p>Scores use recorded evaluation protocols. Changes are absolute percentage points (pp). SE denotes standard error. Subset sample counts appear beside every score.</p>'
    document += _summary(runs) + _plots(runs) + _asset_status(root)
    document += "".join(_run_detail(run, index) for index, run in enumerate(runs))
    document += "</body></html>"
    destination = root / "reports" / "index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="report-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(document)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination
