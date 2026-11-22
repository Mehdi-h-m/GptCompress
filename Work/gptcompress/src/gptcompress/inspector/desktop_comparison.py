"""Native dashboard for registered model versions and recorded measurements."""

import tkinter as tk
from collections import Counter
from tkinter import ttk

from . import inspection
from .comparison_data import default_runs, matched_workload, measured, relative_delta, resource_runs, resource_values

INK = "#182438"
MUTED = "#455670"
LINE = "#d3dfeb"
PANEL = "#f8fafd"
COLORS = ("#225c96", "#7850a0")
FONT = "Segoe UI"
TASK_NAMES = {"gsm8k": "GSM8K", "ifeval": "IFEval", "mmlu_pro": "MMLU-Pro", "gpqa": "GPQA"}
METRIC_NAMES = {"exact_match,flexible-extract": "Flexible exact match",
                "prompt_level_strict_acc,none": "Strict prompt accuracy"}


def gib(value):
    if value is None:
        return "Not measured"
    return f"{value / 1024**3:.2f} GiB" if value >= 1024**3 else f"{value / 1024**2:.2f} MiB"


def duration(value):
    if value is None:
        return "Not measured"
    if value < 60:
        return f"{value:.2f} s"
    minutes, seconds = divmod(round(value), 60)
    hours, minutes = divmod(minutes, 60)
    return (f"{hours}h " if hours else "") + f"{minutes}m {seconds}s"


def run_label(run):
    profile = run.get("profile") or "Run"
    device = run.get("device") or "device unknown"
    return f"{profile.title()} · {device.upper()} · {run['id']}"


def text(canvas, x, y, value, *, size=11, bold=False, color=INK, anchor="nw"):
    canvas.create_text(x, y, text=value, font=(FONT, -size, "bold" if bold else "normal"), fill=color, anchor=anchor)


def pair_bars(canvas, values, y, *, formatter, maximum=None, right=None):
    """Plot both values against one zero-based scale."""
    if maximum is None:
        maximum = max((value for value in values if value is not None), default=0)
    width = right if right is not None else canvas.winfo_width()
    start, end = 44, max(54, width - 120)
    for side, value in enumerate(values):
        top = y + side * 25
        tint = COLORS[side]
        text(canvas, 18, top + 8, "AB"[side], color=tint, bold=True, anchor="w")
        if value is not None:
            canvas.create_rectangle(start, top, end, top + 15, fill="#e9eff6", outline="")
            distance = (end - start) * value / maximum if maximum else 0
            if distance:
                canvas.create_rectangle(start, top, start + distance, top + 15, fill=tint, outline="")
            else:
                canvas.create_line(start, top, start, top + 15, fill=tint, width=2)
        else:
            canvas.create_line(start, top + 8, end, top + 8, fill=LINE, dash=(3, 4))
        text(canvas, width - 16, top + 8, formatter(value), color=tint if value is not None else MUTED, anchor="e")


class ComparisonWindow:
    def __init__(self, parent, versions, initial, preferences=None):
        self.window = tk.Toplevel(parent)
        self.window.title("GptCompress · Model Comparison")
        self.window.geometry(f"{min(1220, self.window.winfo_screenwidth() - 40)}x"
                             f"{min(780, self.window.winfo_screenheight() - 105)}")
        self.window.minsize(900, 675)
        self.window.configure(bg="white")
        self.versions = versions
        self.preferences = preferences if preferences is not None else inspection.STORE / "desktop_comparison.json"
        self.score_rows = []
        self.run_lists = ([], [])
        self.versions_in_view = (versions[0], versions[0])
        self.protocols = []
        try:
            saved = inspection.read_json(self.preferences, {})
            if not isinstance(saved, dict):
                saved = {}
        except (ValueError, OSError):
            saved = {}

        heading = tk.Frame(self.window, bg="white")
        heading.pack(fill="x", padx=22, pady=(15, 9))
        tk.Label(heading, text="Model comparison", font=(FONT, 17, "bold"), fg=INK, bg="white").pack(side="left")

        selectors = tk.Frame(self.window, bg="white")
        selectors.pack(fill="x", padx=22, pady=(0, 12))
        names = Counter(version["name"] for version in versions)
        options = [v["name"] if names[v["name"]] == 1 else f"{v['name']} · {v['id'][:8]}" for v in versions]
        self.pair = []
        for i, side in enumerate(("a", "b")):
            if i:
                ttk.Button(selectors, text="⇄ Swap A/B", command=self.swap).pack(side="left", padx=15)
            tk.Label(selectors, text=f"Version {side.upper()}", bg="white", fg=COLORS[i],
                     font=(FONT, 10, "bold")).pack(side="left", padx=(0, 7))
            selector = ttk.Combobox(selectors, values=options, state="readonly", width=30)
            selector.pack(side="left")
            fallback = initial if i == 0 else next((v["id"] for v in versions if v["id"] != initial), initial)
            key = saved.get(side, fallback)
            selector.current(next((index for index, v in enumerate(versions) if v["id"] == key),
                                  next((index for index, v in enumerate(versions) if v["id"] == fallback), 0)))
            selector.bind("<<ComboboxSelected>>", lambda event: self.select_pair())
            self.pair.append(selector)

        self.footer = tk.Label(self.window, bg="white", fg=MUTED, font=(FONT, 9), anchor="w", justify="left")
        self.footer.pack(side="bottom", fill="x", padx=22, pady=(5, 7))
        lower = tk.Frame(self.window, bg="white")
        lower.pack(side="bottom", fill="x", padx=22)
        row = tk.Frame(lower, bg="white")
        row.pack(fill="x", pady=(4, 5))
        tk.Label(row, text="Recorded resources", bg="white", fg=INK, font=(FONT, 11, "bold")).pack(side="left")
        self.run_pickers = []
        for i, side in enumerate(("A", "B")):
            tk.Label(row, text=side, bg="white", fg=COLORS[i], font=(FONT, 10, "bold")).pack(side="left", padx=(14, 5))
            picker = ttk.Combobox(row, state="readonly", width=29)
            picker.pack(side="left")
            picker.bind("<<ComboboxSelected>>", lambda event: self.update_resources())
            self.run_pickers.append(picker)
        self.resources_note = tk.Label(lower, bg="white", fg=MUTED, anchor="w", font=(FONT, 9))
        self.resources_note.pack(fill="x", pady=(1, 6))
        resource_grid = tk.Frame(lower, bg="white")
        resource_grid.pack(fill="x", pady=(0, 7))
        self.resource_panels = [
            self.canvas(resource_grid, index, 129, lambda i=index: self.draw_resource(i)) for index in range(3)
        ]

        upper = tk.Frame(self.window, bg="white")
        upper.pack(fill="both", expand=True, padx=22, pady=(0, 8))
        upper.grid_columnconfigure(0, weight=1)
        upper.grid_columnconfigure(1, weight=2)
        upper.grid_rowconfigure(1, weight=1)
        tk.Label(upper, text="Checkpoint storage", font=(FONT, 11, "bold"), fg=INK,
                 bg="white").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.storage = tk.Canvas(upper, bg=PANEL, highlightbackground=LINE, highlightthickness=1, width=300)
        self.storage.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.storage.bind("<Configure>", lambda event: self.draw_storage())
        header = tk.Frame(upper, bg="white")
        header.grid(row=0, column=1, sticky="ew", pady=(0, 5))
        tk.Label(header, text="Matched benchmark scores", font=(FONT, 11, "bold"), fg=INK, bg="white").pack(side="left")
        self.protocol_picker = ttk.Combobox(header, state="readonly", width=20)
        self.protocol_picker.pack(side="right")
        self.protocol_picker.bind("<<ComboboxSelected>>", lambda event: self.draw_scores(reset=True))
        score_area = tk.Frame(upper, bg="white")
        score_area.grid(row=1, column=1, sticky="nsew")
        self.scores = tk.Canvas(score_area, bg=PANEL, highlightbackground=LINE, highlightthickness=1, width=470)
        track = ttk.Scrollbar(score_area, command=self.scores.yview)
        track.pack(side="right", fill="y")
        self.scores.pack(side="left", fill="both", expand=True)
        self.scores.configure(yscrollcommand=track.set)
        self.scores.bind("<Configure>", lambda event: self.draw_scores())
        self.scores.bind("<MouseWheel>", lambda event: self.scores.yview_scroll(-int(event.delta / 120), "units"))
        self.select_pair((saved.get("run_a"), saved.get("run_b")))

    def canvas(self, parent, index, height, renderer):
        canvas = tk.Canvas(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1, height=height, width=250)
        canvas.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 8, 0))
        parent.grid_columnconfigure(index, weight=1, uniform="resource")
        canvas.bind("<Configure>", lambda event: renderer())
        return canvas

    def exists(self):
        return bool(self.window.winfo_exists())

    def swap(self):
        runs = self.chosen_runs()
        a, b = (selector.current() for selector in self.pair)
        self.pair[0].current(b)
        self.pair[1].current(a)
        self.select_pair(tuple(run["id"] if run else None for run in reversed(runs)))

    def select_pair(self, preferred=(None, None)):
        self.versions_in_view = tuple(self.versions[selector.current()] for selector in self.pair)
        a, b = self.versions_in_view
        self.score_rows = inspection.benchmark_deltas(a, b)
        self.protocols = list(dict.fromkeys(row["protocol"] for row in self.score_rows))
        names = []
        for protocol in self.protocols:
            profile = next(row["profile"] for row in self.score_rows if row["protocol"] == protocol)
            names.append(f"{(profile or 'Run').title()} · {protocol[:8]}")
        self.protocol_picker.configure(values=names, state="readonly" if names else "disabled")
        if names:
            self.protocol_picker.current(0)
        else:
            self.protocol_picker.set("")

        self.run_lists = tuple(resource_runs(version) for version in self.versions_in_view)
        defaults = default_runs(*self.run_lists)
        for index, (picker, runs, default) in enumerate(zip(self.run_pickers, self.run_lists, defaults)):
            picker.configure(values=[run_label(run) for run in runs], state="readonly" if runs else "disabled")
            if runs:
                key = preferred[index] or default["id"]
                picker.current(next((i for i, run in enumerate(runs) if run["id"] == key), runs.index(default)))
            else:
                picker.set("No measured runs")
        self.draw_storage()
        self.draw_scores(reset=True)
        self.update_resources()

    def chosen_runs(self):
        return tuple(runs[picker.current()] if picker.current() >= 0 else None
                     for runs, picker in zip(self.run_lists, self.run_pickers))

    def update_resources(self):
        a, b = self.chosen_runs()
        if matched_workload(a, b):
            note = "Matched resource runs · identical benchmark questions, protocol, device and CPU threads."
        elif a and b and a.get("profile") == b.get("profile") == "smoke":
            tokens = tuple(run.get("smoke_tokens") for run in (a, b))
            note = "Separate smoke measurements"
            if all(count is not None for count in tokens):
                note += f" · A generated {tokens[0]} tokens · B generated {tokens[1]} tokens"
            note += "."
        else:
            note = "Resources show the selected runs. Matching measurements are needed for deltas."
        self.resources_note.configure(text=note)
        for index in range(3):
            self.draw_resource(index)
        ids = [f"{side} run {run['id']}" for side, run in zip("AB", (a, b)) if run]
        message = "\n".join(filter(None, ("  ·  ".join(ids),
                                        "Elapsed wall time · peak process working set · peak CUDA memory reserved by PyTorch")))
        try:
            first, second = self.versions_in_view
            inspection.save_json(self.preferences, {
                "a": first["id"], "b": second["id"],
                "run_a": a["id"] if a else None, "run_b": b["id"] if b else None
            })
        except OSError:
            message = "Comparison selections could not be saved to disk."
        self.footer.configure(text=message)

    def draw_storage(self):
        chart = self.storage
        chart.delete("all")
        values = tuple(measured(version.get("bytes")) for version in self.versions_in_view)
        text(chart, 16, 16, "Serialized checkpoint size", size=14, bold=True)
        pair_bars(chart, values, 51, formatter=gib)
        delta = relative_delta(*values)
        if delta is not None:
            text(chart, 16, 119, f"{delta:+.1f}% relative to Version A", color=MUTED)
        text(chart, 16, 158, "Stored parameters", size=14, bold=True)
        params = tuple(measured(version.get("parameters")) for version in self.versions_in_view)
        reported = lambda value: f"{value / 1e9:.3f} B" if value is not None else "Not measured"
        pair_bars(chart, params, 191, formatter=reported)

    def draw_resource(self, index):
        chart = self.resource_panels[index]
        chart.delete("all")
        title, formatter = (("Elapsed time", duration), ("Peak RAM", gib), ("Peak VRAM", gib))[index]
        runs = self.chosen_runs()
        values = tuple(resource_values(run)[index] for run in runs)
        text(chart, 16, 13, title, size=14, bold=True)
        pair_bars(chart, values, 42, formatter=formatter)
        delta = relative_delta(*values) if matched_workload(*runs) else None
        if delta is not None:
            text(chart, 16, 103, f"{delta:+.1f}% relative to Version A", color=MUTED)

    def draw_scores(self, reset=False):
        chart = self.scores
        chart.delete("all")
        protocol = self.protocol_picker.current()
        rows = [row for row in self.score_rows if self.protocols[protocol] == row["protocol"]] if protocol >= 0 else []
        width = chart.winfo_width()
        chart.configure(scrollregion=(0, 0, width, max(chart.winfo_height(), 39 + len(rows) * 111)))
        if reset:
            chart.yview_moveto(0)
        if not rows:
            text(chart, 16, 16, "No completed benchmark pair matches the same protocol and questions.", color=MUTED)
            return
        text(chart, 16, 12, "Completed scores · identical protocol, metric and question IDs", color=MUTED)
        for index, row in enumerate(rows):
            top = 41 + index * 111
            name = TASK_NAMES.get(row["task"], row["task"])
            text(chart, 16, top, f"{name}  ·  {row['n']} questions per version", size=13, bold=True)
            text(chart, width - 16, top + 2, f"{row['delta_pp']:+g} pp (B − A)", color=MUTED, anchor="ne")
            metric = METRIC_NAMES.get(row["metric"], row["metric"])
            text(chart, 16, top + 18, metric, color=MUTED)
            pair_bars(chart, (row["a"], row["b"]), top + 39,
                      formatter=lambda value: f"{value:.1%}", maximum=1.0)

