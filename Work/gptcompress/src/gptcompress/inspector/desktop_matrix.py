"""Native Sprint 2 matrix viewer and asynchronous chart rendering."""

import math
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk
from tkinter.font import Font

from .matrix import inspect_slice, np, projection_sources

INK = "#182438"
MUTED = "#455670"
LINE = "#b7c5d5"
ACCENT = "#236653"
EMPTY = "#f8fafd"
FONT = "Segoe UI"
SIZES = (512, 256, 128, 64, 32, 16, 8)


def fmt(value):
    return "—" if value is None or not math.isfinite(value) else f"{value:.4g}"


class MatrixWindow:
    def __init__(self, parent, versions, version, module):
        self.window = tk.Toplevel(parent)
        self.window.title("GptCompress · Matrix Inspector")
        self.window.geometry(f"{min(1190, self.window.winfo_screenwidth() - 50)}x"
                             f"{min(790, self.window.winfo_screenheight() - 105)}")
        self.window.minsize(950, 550)
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.versions = versions
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.requested = None
        self.timer = None
        self.result = None
        self.display_request = None
        self.image = None
        self.map_bounds = None
        self.sources = []
        self.module = module
        self.tick_font = Font(self.window, family=FONT, size=8)

        top = tk.Frame(self.window, bg="white")
        top.pack(fill="x", padx=20, pady=(13, 5))
        self.title = tk.Label(top, bg="white", fg=INK, font=(FONT, 16, "bold"), anchor="w")
        self.title.pack(side="left")
        self.version_select = ttk.Combobox(top, state="readonly", width=22, values=[v["name"] for v in versions])
        self.version_select.pack(side="right")
        self.version_select.bind("<<ComboboxSelected>>", lambda event: self.load_sources(preserve_origin=True))

        toolbar = tk.Frame(self.window, bg="white")
        toolbar.pack(fill="x", padx=20, pady=(5, 0))
        self.source_select = ttk.Combobox(toolbar, state="readonly", width=21)
        self.source_select.pack(side="left", padx=(0, 12))
        self.source_select.bind("<<ComboboxSelected>>", lambda event: self.reset_slice())
        self.row = tk.StringVar(self.window, "0")
        self.col = tk.StringVar(self.window, "0")
        self.side = tk.StringVar(self.window, "512")
        self.spectrum = tk.BooleanVar(self.window, False)
        for label, value in (("Row start", self.row), ("Column start", self.col)):
            tk.Label(toolbar, text=label, fg=MUTED, bg="white").pack(side="left")
            entry = ttk.Entry(toolbar, textvariable=value, width=7)
            entry.pack(side="left", padx=(3, 10))
            entry.bind("<Return>", lambda event: self.request())
        tk.Label(toolbar, text="Slice size", fg=MUTED, bg="white").pack(side="left")
        self.size = ttk.Combobox(toolbar, width=6, textvariable=self.side, state="readonly", values=SIZES)
        self.size.pack(side="left", padx=(3, 8))
        self.size.bind("<<ComboboxSelected>>", lambda event: self.request())
        ttk.Button(toolbar, text="View slice", command=self.request).pack(side="left")
        self.window.option_add("*TCheckbutton.padding", 5)
        ttk.Checkbutton(toolbar, text="Singular values", variable=self.spectrum,
                        command=self.request).pack(side="left", padx=8)
        self.meta = tk.Label(self.window, bg="white", fg=MUTED, font=(FONT, 10), anchor="w")
        self.meta.pack(fill="x", padx=20, pady=(7, 4))
        self.status = tk.Label(self.window, bg="white", fg=ACCENT, font=(FONT, 10), anchor="w")
        self.status.pack(side="bottom", fill="x", padx=20, pady=(3, 7))

        area = tk.Frame(self.window, bg="white")
        area.pack(fill="both", expand=True, padx=20, pady=(3, 5))
        area.columnconfigure(0, weight=11, uniform="chart")
        area.columnconfigure(1, weight=10, uniform="chart")
        area.rowconfigure(0, weight=1)
        left = tk.Frame(area, bg="white")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.heat = tk.Canvas(left, background=EMPTY, highlightthickness=1, highlightbackground=LINE)
        self.heat.pack(fill="both", expand=True)
        self.heat.bind("<Configure>", lambda event: self.draw_heatmap())
        self.heat.bind("<Motion>", self.probe)
        self.heat.bind("<Button-1>", lambda event: self.zoom(event, 1))
        self.heat.bind("<MouseWheel>", lambda event: self.zoom(event, 1 if event.delta > 0 else -1))
        self.summary = tk.Label(left, bg="white", fg=INK, justify="left", anchor="w", font=(FONT, 10))
        self.summary.pack(fill="x", pady=(7, 0))
        right = tk.Frame(area, bg="white")
        right.grid(row=0, column=1, sticky="nsew")
        self.plots = {}
        for index, title in enumerate(("Histogram", "Row norms", "Column norms", "Singular values")):
            plot = tk.Canvas(right, background=EMPTY, highlightthickness=1, highlightbackground=LINE)
            row, col = divmod(index, 2)
            plot.grid(row=row, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0),
                      pady=(0 if row == 0 else 6, 0))
            right.rowconfigure(row, weight=1, uniform="plots")
            right.columnconfigure(col, weight=1, uniform="plots")
            plot.bind("<Configure>", lambda event, name=title: self.draw_plot(name))
            self.plots[title] = plot
        self.open(version, module)

    def exists(self):
        return bool(self.window.winfo_exists())

    def open(self, version, module):
        self.module = module
        self.version_select.current(next(i for i, entry in enumerate(self.versions) if entry["id"] == version["id"]))
        self.title.configure(text=module)
        self.load_sources()
        self.window.lift()

    def load_sources(self, preserve_origin=False):
        self.result = self.display_request = self.requested = None
        self.sources = []
        try:
            self.sources = projection_sources(self.versions[self.version_select.current()], self.module)
        except (OSError, ValueError) as error:
            self.status.configure(text=str(error))
            self.source_select.configure(values=())
            self.source_select.set("")
            self.meta.configure(text="")
            self.draw()
            return
        self.source_select.configure(values=[source.label for source in self.sources])
        if not self.sources:
            self.source_select.set("")
            self.meta.configure(text="This projection is unavailable in the selected model.")
            self.status.configure(text="")
            self.draw()
            return
        self.source_select.current(0)
        if preserve_origin:
            self.request()
        else:
            self.reset_slice()

    def reset_slice(self):
        self.row.set("0")
        self.col.set("0")
        self.request()

    def request(self):
        index = self.source_select.current()
        if index < 0 or index >= len(self.sources):
            return
        source = self.sources[index]
        try:
            row, col, side = int(self.row.get()), int(self.col.get()), int(self.side.get())
            if not (0 <= row < source.shape[0] and 0 <= col < source.shape[1]):
                raise ValueError(f"Enter a row from 0 to {source.shape[0] - 1:,} and a column from 0 to {source.shape[1] - 1:,}.")
        except ValueError as error:
            self.status.configure(text=str(error))
            return
        self.requested = (source, row, col, side, self.spectrum.get())
        self.result = self.display_request = None
        self.meta.configure(text=f"{source.shape[0]:,} × {source.shape[1]:,}  ·  {source.representation}")
        self.status.configure(text="Reading slice and calculating statistics…")
        self.draw()
        if self.future is None:
            self.start_requested()

    @staticmethod
    def compute(requested):
        requested[0].validate()
        return inspect_slice(*requested)

    def start_requested(self):
        if self.requested is None:
            return
        self.processing = self.requested
        self.future = self.executor.submit(self.compute, self.processing)
        self.timer = self.window.after(30, self.poll)

    def poll(self):
        self.timer = None
        if not self.future.done():
            self.timer = self.window.after(30, self.poll)
            return
        try:
            if self.processing == self.requested:
                self.result = self.future.result()
                self.display_request = self.processing
                self.status.configure(text="Hover for coordinates · Click or scroll to zoom the slice")
                self.draw()
        except (OSError, ValueError, np.linalg.LinAlgError) as error:
            self.status.configure(text=f"Slice unavailable · {error}")
        finally:
            self.future = None
        if self.processing != self.requested:
            self.start_requested()

    def close(self):
        if self.timer is not None:
            self.window.after_cancel(self.timer)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()

    def draw(self):
        self.draw_heatmap()
        for name in self.plots:
            self.draw_plot(name)
        if not self.result:
            self.summary.configure(text="")
            return
        stats = self.result["statistics"]
        self.summary.configure(text=(
            f"Min {fmt(stats['min'])}     Max {fmt(stats['max'])}     Mean {fmt(stats['mean'])}     "
            f"Std. dev. {fmt(stats['std'])}\nFrobenius norm {fmt(stats['norm'])}     Nonfinite {stats['nonfinite']:,}"))

    def draw_heatmap(self):
        canvas = self.heat
        canvas.delete("all")
        self.map_bounds = None
        canvas.create_text(13, 14, text="Weight heatmap", anchor="nw", fill=INK, font=(FONT, 11, "bold"))
        if not self.result:
            return
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 125 or height < 120:
            return
        x, y, map_width, map_height = 53, 67, width - 74, height - 129
        if map_height < 20:
            return
        values = self.result["heatmap"]
        finite = np.isfinite(values)
        limit = float(np.max(np.abs(values[finite]))) if finite.any() else 0
        strength = np.where(finite, np.abs(values) / (limit or 1), 0)[..., None]
        positive = values[..., None] >= 0
        neutral = np.array([250, 250, 247])
        ends = np.where(positive, [176, 50, 25], [24, 84, 170])
        rgb = np.where(finite[..., None], neutral + strength * (ends - neutral), [146, 140, 150]).astype(np.uint8)
        iy = np.arange(map_height) * rgb.shape[0] // map_height
        ix = np.arange(map_width) * rgb.shape[1] // map_width
        pixels = rgb[iy[:, None], ix].tobytes()
        self.image = tk.PhotoImage(master=canvas,
                                   data=f"P6\n{map_width} {map_height}\n255\n".encode() + pixels, format="PPM")
        canvas.create_image(x, y, image=self.image, anchor="nw")
        canvas.create_rectangle(x, y, x + map_width, y + map_height, outline=LINE)
        _, row, col, _, _ = self.display_request
        count_r, count_c = self.result["shape"]
        for px, py, anchor, label in ((x, y - 5, "sw", f"Column {col:,}"),
                                       (x + map_width, y - 5, "se", f"{col + count_c - 1:,}"),
                                       (x - 6, y, "ne", f"{row:,}"),
                                       (x - 6, y + map_height, "se", f"{row + count_r - 1:,}")):
            canvas.create_text(px, py, anchor=anchor, text=label, fill=MUTED, font=(FONT, 9))
        self.legend(canvas, x, y + map_height + 17, map_width, limit)
        step = self.result["step"]
        note = "One cell per weight" if step == 1 else f"Each cell shows the signed largest magnitude in up to {step} × {step} weights"
        canvas.create_text(x, height - 18, text=note, fill=MUTED, anchor="w", font=(FONT, 8))
        self.map_bounds = (x, y, map_width, map_height)

    @staticmethod
    def legend(canvas, x, y, width, limit):
        canvas.create_rectangle(x, y, x + 13, y + 10, fill="#1854aa", outline="")
        canvas.create_text(x + 18, y + 5, text=f"−{fmt(limit)}", fill=MUTED, anchor="w", font=(FONT, 8))
        canvas.create_text(x + width / 2, y + 5, text="0", fill=MUTED, font=(FONT, 8))
        canvas.create_rectangle(x + width - 13, y, x + width, y + 10, fill="#b03219", outline="")
        canvas.create_text(x + width - 18, y + 5, text=f"+{fmt(limit)}", fill=MUTED, anchor="e", font=(FONT, 8))

    def heat_cell(self, event):
        if not self.result or not self.map_bounds:
            return None
        x, y, width, height = self.map_bounds
        values = self.result["heatmap"]
        if not (x <= event.x < x + width and y <= event.y < y + height):
            return None
        return (event.y - y) * values.shape[0] // height, (event.x - x) * values.shape[1] // width

    def probe(self, event):
        point = self.heat_cell(event)
        if point is None:
            return
        r, c = point
        _, row, col, _, _ = self.display_request
        step = self.result["step"]
        end_r, end_c = self.result["shape"]
        label = (f"Row {row + r * step:,} · Column {col + c * step:,}" if step == 1 else
                 f"Rows {row + r * step:,}–{row + min((r + 1) * step, end_r) - 1:,} · "
                 f"Columns {col + c * step:,}–{col + min((c + 1) * step, end_c) - 1:,}")
        self.status.configure(text=label + f" · {fmt(float(self.result['heatmap'][r, c]))}")

    def zoom(self, event, direction):
        point = self.heat_cell(event)
        if point is None:
            return
        source, row, col, side, _ = self.display_request
        index = max(0, min(len(SIZES) - 1, SIZES.index(side) + direction))
        new_side = SIZES[index]
        if side == new_side:
            return "break"
        step = self.result["step"]
        for value, start, cell, extent in zip((self.row, self.col), (row, col), point, source.shape):
            value.set(str(max(0, min(max(0, extent - new_side), start + cell * step - new_side // 2))))
        self.side.set(str(new_side))
        self.request()
        return "break"

    def draw_plot(self, title):
        plot = self.plots[title]
        plot.delete("all")
        plot.create_text(13, 14, text=title, anchor="nw", fill=INK, font=(FONT, 11, "bold"))
        if not self.result:
            return
        bars = title == "Histogram"
        if bars:
            values, axis = self.result["histogram"]
            start, end = map(fmt, (float(axis[0]), float(axis[-1])))
        elif title == "Singular values":
            values = self.result["spectrum"]
            if values is None:
                text = ("Unavailable · nonfinite weights" if self.result["spectrum_unavailable"] else
                        "Enable singular values above")
                plot.create_text(13, 45, text=text, width=max(70, plot.winfo_width() - 26),
                                 anchor="nw", fill=MUTED, font=(FONT, 9))
                return
            start, end = "1", f"{len(values):,}"
        else:
            row_plot = title == "Row norms"
            values = self.result["row_norms" if row_plot else "column_norms"]
            origin = self.display_request[1 if row_plot else 2]
            start, end = f"{origin:,}", f"{origin + len(values) - 1:,}"
        width, height = plot.winfo_width(), plot.winfo_height()
        max_value = float(np.max(values))
        left = max(34, self.tick_font.measure(fmt(max_value)) + 12)
        right, top, bottom = width - 13, 57, height - 26
        if right < left + 20 or bottom < top + 15:
            return
        maximum = max_value or 1.0
        plot.create_line(left, top, left, bottom, right, bottom, fill=LINE)
        plot.create_text(left - 4, top, text=fmt(max_value), anchor="e", fill=MUTED, font=self.tick_font)
        plot.create_text(left, bottom + 7, text=start, anchor="nw", fill=MUTED, font=(FONT, 8))
        plot.create_text(right, bottom + 7, text=end, anchor="ne", fill=MUTED, font=(FONT, 8))
        interval = (right - left) / len(values)
        points = []
        for i, value in enumerate(values):
            x = left + (i + .5) * interval
            y = bottom - (bottom - top) * float(value) / maximum
            if bars:
                plot.create_rectangle(x - .42 * interval, y, x + .42 * interval, bottom, fill="#557cb4", outline="")
            else:
                points.extend((x, y))
        if len(points) >= 4:
            plot.create_line(*points, fill=ACCENT, width=1.6)
