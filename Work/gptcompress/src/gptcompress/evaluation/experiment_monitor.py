"""Read-only desktop monitor for local GptCompress experiments."""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
import webbrowser

import psutil


ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ((256, "mlp", 108), (256, "attention", 144), (256, "combined", 252),
            (128, "combined", 252), (512, "combined", 252))
TASKS = ("gsm8k", "ifeval")


@dataclass
class Task:
    name: str
    status: str
    progress: str = ""
    details: str = ""


def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def read_json(path):
    try:
        return json.loads(read_text(path))
    except ValueError:
        return {}


def active_commands():
    commands = []
    for process in psutil.process_iter(["name"]):
        if (process.info["name"] or "").casefold() not in ("python.exe", "pythonw.exe", "python", "powershell.exe"):
            continue
        try:
            commands.append(" ".join(process.cmdline()).replace("\\", "/").casefold())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return commands


def stage_status(record, active):
    status = record.get("status", "Pending")
    return "Interrupted" if status == "running" and not active else status.capitalize()


def conversion(name, folder, total, active):
    state = read_json(folder / "progress.json")
    if not state:
        return Task(name, "Pending")
    rows = state.get("matrices", [])
    done = len(rows)
    details = state.get("error", "")
    if state.get("status") == "complete":
        manifest = read_json(folder / "manifest.json")
        if manifest.get("files"):
            total_bytes = sum(record["bytes"] for record in manifest["files"])
            total_bytes += (folder / "manifest.json").stat().st_size
            details = f"{total_bytes / 1e9:.2f} GB checkpoint"
    elif rows and not details:
        details = f"Last recorded matrix: {rows[-1].get('name', 'unknown')}"
    return Task(name, stage_status(state, active), f"{done}/{total} matrices", details)


def generation_progress(log):
    """Read generation counters only, skipping the harness's setup/cache counters."""
    task = ""
    progress = {}
    for line in log.splitlines():
        match = re.search(r"Evaluating examples for tasks \['(gsm8k|ifeval)'\]", line)
        if match:
            task = match[1]
        counter = re.search(r"Running generate_until requests:.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]", line)
        if task and counter:
            done, total, timing = counter.groups()
            progress[task] = (int(done), int(total), timing.split(",")[0])
    return progress


def benchmark(name, directory, log, active, queued=False):
    if directory is None:
        return Task(name, "Queued" if queued else "Pending")
    metrics = read_json(directory / "metrics.json")
    if not metrics:
        return Task(name, "Starting" if active else "Interrupted")
    status = stage_status(metrics, active)
    if "smoke" in metrics:
        answer = metrics["smoke"].get("response", "").strip()
        return Task(name, status, "Generation recorded", f"Response: {answer!r}")
    progress = generation_progress(log)
    counts, details = [], []
    for key in TASKS:
        result = metrics.get("tasks", {}).get(key, {})
        if result.get("status") == "complete":
            counts.append(f"{key.upper()} {result['n']}/{result['n']}")
            details.append(f"{key.upper()} score {result['score']:.1%}")
        elif key in progress:
            done, total, timing = progress[key]
            counts.append(f"{key.upper()} {done}/{total}")
            details.append(f"{key.upper()} elapsed/remaining estimate {timing}" if active else key.upper())
        elif key in metrics.get("tasks", {}):
            counts.append(f"{key.upper()} {result.get('status', 'pending')}")
        if result.get("error"):
            details.append(result["error"])
    if metrics.get("error"):
        details.append(metrics["error"])
    return Task(name, status, "  ·  ".join(counts), "  ·  ".join(details))


def snapshot():
    commands = active_commands()
    queue_running = any("/als-001-dev.ps1" in cmd for cmd in commands)
    queues = sorted((ROOT / ".cache").glob("als-001-dev-*/status.txt"))
    queue = queues[-1].parent if queues else None
    runs = {}
    for path in sorted((ROOT / "experiments").glob("*/manifest.json")):
        record = read_json(path)
        compression = record.get("compression") or {}
        if record.get("profile"):
            runs[(record["profile"], compression.get("rank"), compression.get("target"))] = path.parent

    def evaluate(name, profile, rank=None, target=None, queue_step=None):
        log = read_text(queue / f"{queue_step}.log.stderr") if queue and queue_step else ""
        marker = re.search(r"INFO:gptcompress:Run (\S+)", log)
        run = ROOT / "experiments" / marker[1] if marker else runs.get((profile, rank, target))
        if run and not log:
            log = read_text(run / "execution.log")
        model = f"als-r{rank}-{target}" if rank else None
        flag = "--smoke" if profile == "smoke" else f"--profile {profile}"
        active = any("src/gptcompress/evaluation/run_evaluation.py" in cmd and flag in cmd and
                     (model in cmd if model else "--model" not in cmd) for cmd in commands)
        if run is None and active:
            return Task(name, "Starting")
        return benchmark(name, run, log, active, queue_running and bool(queue_step))

    rows = [evaluate("Original pilot", "pilot")]
    rows.append(conversion("Rank-256 weight probe", ROOT / "experiments/ALS-001-probe-r256", 6,
                           any("src/gptcompress/compression/compress_model.py" in cmd and "--probe" in cmd for cmd in commands)))
    for rank, target, count in VARIANTS:
        folder = f"als-r{rank}-{target}"
        active = any("src/gptcompress/compression/compress_model.py" in cmd and folder in cmd for cmd in commands)
        rows.append(conversion(f"Convert rank {rank} · {target.upper()}", ROOT / "models/compressed" / folder,
                               count, active))
    rows.append(evaluate("Compressed inference check", "smoke", 256, "mlp", "compressed-smoke"))
    rows.append(evaluate("Original development", "dev", queue_step="original-dev"))
    for rank, target, _ in VARIANTS:
        step = "mlp-dev" if (rank, target) == (256, "mlp") else None
        rows.append(evaluate(f"Development · rank {rank} {target.upper()}", "dev", rank, target, step))
    rows.append(evaluate("Original held-out", "heldout"))
    heldout = [(key[1], key[2]) for key in runs if key[0] == "heldout" and key[1] is not None]
    for rank, target in heldout:
        rows.append(evaluate(f"Held-out · rank {rank} {target.upper()}", "heldout", rank, target))
    if not heldout:
        rows.append(Task("Selected variant held-out", "Pending", details="Awaiting development results and selection"))
    return rows


def window():
    root = tk.Tk()
    root.title("GptCompress · Experiment progress")
    root.geometry("1080x690")
    root.minsize(720, 440)
    root.configure(background="#edf2f8")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#edf2f8")
    style.configure("TLabel", background="#edf2f8", foreground="#152741", font=("Segoe UI", 10))
    style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
    style.configure("Treeview", rowheight=28, font=("Segoe UI", 10), fieldbackground="#ffffff")
    style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#d9e5f3")
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    header = ttk.Frame(frame)
    header.pack(fill="x", pady=(0, 12))
    ttk.Label(header, text="Experiment progress", style="Title.TLabel").pack(side="left")
    ttk.Button(header, text="Open benchmark report",
               command=lambda: webbrowser.open((ROOT / "reports/index.html").as_uri())).pack(side="right")
    message = tk.StringVar(root)
    ttk.Label(frame, textvariable=message).pack(anchor="w", pady=(0, 8))
    body = ttk.Frame(frame)
    body.pack(fill="both", expand=True)
    tree = ttk.Treeview(body, columns=("task", "status", "progress"), show="headings", selectmode="browse")
    for key, label, width in (("task", "Task", 335), ("status", "Status", 120),
                              ("progress", "Recorded progress", 365)):
        tree.heading(key, text=label)
        tree.column(key, width=width, minwidth=95, stretch=key != "status")
    scrollbar = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    for tag, color in (("Complete", "#126847"), ("Running", "#145bbb"), ("Failed", "#9b332d"),
                       ("Interrupted", "#8d4b18"), ("Queued", "#705887")):
        tree.tag_configure(tag, foreground=color)
    detail = tk.StringVar(root, "Select a task to see its status details.")
    ttk.Label(frame, textvariable=detail, wraplength=900, justify="left").pack(fill="x", pady=(12, 0))
    tasks = {}

    def show_detail(_event=None):
        selected = tree.selection()
        if selected and selected[0] in tasks:
            task = tasks[selected[0]]
            detail.set(f"{task.name} — {task.details or task.status}")

    def refresh():
        try:
            for task in snapshot():
                tasks[task.name] = task
                values = (task.name, task.status, task.progress)
                if tree.exists(task.name):
                    tree.item(task.name, values=values, tags=(task.status,))
                else:
                    tree.insert("", "end", iid=task.name, values=values, tags=(task.status,))
            message.set(f"Updated {datetime.now():%H:%M:%S}  ·  Refreshes every 5 seconds")
            show_detail()
        except OSError as error:
            message.set(f"Could not read experiment records. Retrying in 5 seconds. {error}")
        root.after(5000, refresh)

    tree.bind("<<TreeviewSelect>>", show_detail)
    refresh()
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true", help="Print the current task list without opening a window")
    if parser.parse_args().snapshot:
        for row in snapshot():
            print(f"{row.name} | {row.status} | {row.progress} | {row.details}")
    else:
        window().mainloop()
