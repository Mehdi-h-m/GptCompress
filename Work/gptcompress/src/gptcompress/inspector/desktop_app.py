"""GptCompress native architecture, matrix, and model comparison explorer."""

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gptcompress.inspector import inspection

BG = "#ffffff"
PANEL = "#f6f8fc"
NODE = "#f3f6fa"
EDGE = "#b7c5d5"
TEXT = "#182438"
MUTED = "#455670"
TEAL = "#236653"
WIRE = "#374764"
COLORS = {
    "attention": ("#fff0df", "#c78043"),
    "mlp": ("#e9f4ff", "#6194c3"),
    "norm": ("#fff9dc", "#b9983e"),
    "embedding": ("#fbeaf1", "#bc7c9c"),
    "output": ("#e8f6ee", "#6b9c7d"),
    "factor": ("#f0edff", "#8d79c4"),
}
FONT = "Segoe UI"

LABELS = {
    "embed_tokens": "Token embedding", "self_attn": "Self-attention", "mlp": "Feed-forward network",
    "input_layernorm": "Input RMSNorm", "post_attention_layernorm": "Post-attention RMSNorm",
    "norm": "Final RMSNorm", "gate_proj": "Gate projection", "up_proj": "Up projection",
    "down_proj": "Down projection", "q_proj": "Query projection", "k_proj": "Key projection",
    "v_proj": "Value projection", "o_proj": "Output projection",
    "q_norm": "Query RMSNorm", "k_norm": "Key RMSNorm",
}
# Presentation-only lookups. Every drawn element resolves names against the version's own module tree.
KIND_TONES = {"normalization": "norm", "self-attention": "attention", "feed-forward network": "mlp",
              "factorized": "factor"}
RESIDUAL_WORDS = {"self-attention": "attention", "feed-forward network": "MLP"}


def module_label(name):
    tail = name.rsplit(".", 1)[-1]
    return f"Block {tail}" if tail.isdecimal() else LABELS.get(tail, tail)


def shape_text(shape):
    return " × ".join(f"{width:,}" for width in shape)


def size_text(count):
    if count >= 1024**3:
        return f"{count / 1024**3:.2f} GiB"
    if count >= 1024**2:
        return f"{count / 1024**2:.2f} MiB"
    return f"{count:,} bytes"


class ArchitectureWindow:
    def __init__(self, root, versions, comparison_versions=None):
        self.root = root
        self.versions = versions
        self.comparison_versions = comparison_versions if comparison_versions is not None else versions
        self.version_index = 0
        self.current = ""
        self.block_index = 0
        self.scale = 1.0
        self.extent = (880, 610)
        self.press = None
        self.module_index = {}
        self.children = {}
        root.title("GptCompress · Model Inspector")
        root.configure(background=BG)
        root.geometry("1240x850")
        root.minsize(760, 600)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=NODE, foreground=TEXT, background=EDGE,
                        selectbackground=NODE, selectforeground=TEXT, padding=7)
        style.map("TCombobox", fieldbackground=[("readonly", NODE)], foreground=[("readonly", TEXT)])
        style.configure("TButton", background=NODE, foreground=TEXT, padding=(12, 7),
                        bordercolor=EDGE, font=(FONT, 10))
        style.map("TButton", background=[("active", "#e5edff")], foreground=[("disabled", MUTED)])
        style.configure("TScrollbar", background=EDGE, troughcolor=PANEL, arrowcolor=MUTED)
        root.option_add("*TCombobox*Listbox.background", NODE)
        root.option_add("*TCombobox*Listbox.foreground", TEXT)

        header = tk.Frame(root, bg=PANEL, padx=22, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="GptCompress", font=(FONT, 17, "bold"), bg=PANEL, fg=TEXT).pack(side="left")
        tk.Label(header, text="MODEL INSPECTOR", font=(FONT, 9, "bold"), bg=PANEL, fg=TEAL,
                 padx=18).pack(side="left")
        selector = ttk.Combobox(header, state="readonly", width=28,
                                values=[version["name"] for version in versions])
        selector.pack(side="right")
        ttk.Button(header, text="Compare models", command=self.open_comparison).pack(side="right", padx=(0, 15))
        selector.current(0)
        selector.bind("<<ComboboxSelected>>", lambda event: self.select_version(selector.current()))

        controls = tk.Frame(root, bg=BG, padx=20, pady=9)
        controls.pack(fill="x")
        self.back = ttk.Button(controls, text="‹ Back", command=self.go_back)
        self.back.pack(side="left")
        self.breadcrumbs = tk.Frame(controls, bg=BG, padx=10)
        self.breadcrumbs.pack(side="left", fill="x", expand=True)
        ttk.Button(controls, text="+", width=3, command=lambda: self.zoom(1.18)).pack(side="right")
        self.zoom_label = ttk.Button(controls, width=7, command=self.reset_zoom)
        self.zoom_label.pack(side="right", padx=4)
        ttk.Button(controls, text="−", width=3, command=lambda: self.zoom(1 / 1.18)).pack(side="right")

        viewport = tk.Frame(root, bg=EDGE, padx=1, pady=1)
        viewport.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(viewport, bg=BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        for direction in ("x", "y"):
            horizontal = direction == "x"
            scrollbar = ttk.Scrollbar(viewport, orient="horizontal" if horizontal else "vertical",
                                      command=self.canvas.xview if horizontal else self.canvas.yview)
            scrollbar.grid(row=1 if horizontal else 0, column=0 if horizontal else 1,
                           sticky="ew" if horizontal else "ns")
            self.canvas.configure(**{direction + "scrollcommand": scrollbar.set})
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-1>", self.finish_pan)
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<Control-MouseWheel>", self.zoom_wheel)
        self.canvas.bind("<Motion>", self.track_hover)
        self.canvas.bind("<Leave>", lambda event: self.set_hover(None))
        root.bind("<Escape>", lambda event: self.go_back())
        root.bind("<Left>", lambda event: self.key_step(event, -1))
        root.bind("<Right>", lambda event: self.key_step(event, 1))
        self.status = tk.Label(root, text="Scroll over the decoder to change blocks · Ctrl+scroll to zoom · Drag to pan",
                               bg=BG, fg=MUTED, font=(FONT, 10), anchor="w")
        self.status.pack(fill="x", padx=21, pady=(0, 9))
        self.actions = {}
        self.hover_nodes = {}
        self.hover_levels = {}
        self.hovered = None
        self.hover_timer = None
        self.block_bounds = None
        self.matrix_window = None
        self.comparison_window = None
        self.select_version(0)

    def select_version(self, index):
        self.version_index = index
        self.version = self.versions[index]
        self.config = inspection.read_json(Path(self.version["path"]) / "config.json", {})
        self.module_index = {module["name"]: module for module in self.version["modules"]}
        self.blocks = sorted((name for name, module in self.module_index.items()
                              if module["kind"] == "transformer block"),
                             key=lambda name: int(name.rsplit(".", 1)[-1]))
        self.block_index = min(self.block_index, max(0, len(self.blocks) - 1))
        self.children = {}
        for name in self.module_index:
            parent = name.rpartition(".")[0]
            self.children.setdefault(parent, []).append(name)
        self.open_module(self.current if self.current in self.module_index else "")

    def open_module(self, name):
        self.current = name
        depth = len(self.blocks[0].split(".")) if self.blocks else 3
        block = ".".join(name.split(".")[:depth])
        if block in self.blocks:
            self.block_index = self.blocks.index(block)
        self.scale = 1
        self.render()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def parent(self, name):
        parent = name.rpartition(".")[0]
        while parent and parent not in self.module_index:
            parent = parent.rpartition(".")[0]
        return parent

    def go_back(self):
        if self.current:
            self.open_module(self.parent(self.current))

    def start_pan(self, event):
        self.press = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        if self.press and self.travel(event) > 5:
            self.set_hover(None)
            self.canvas.configure(cursor="fleur")
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def travel(self, event):
        return abs(event.x - self.press[0]) + abs(event.y - self.press[1])

    def finish_pan(self, event):
        if self.press and self.travel(event) <= 5:
            key = self.hit_node(event.x, event.y)
            if key is not None:
                target = self.actions[key]
                if isinstance(target, int):
                    self.step_block(target)
                elif callable(target):
                    target()
                else:
                    self.open_module(target)
        self.press = None
        self.track_hover(event)

    def hit_node(self, x, y):
        x, y = self.canvas.canvasx(x), self.canvas.canvasy(y)
        for item in reversed(self.canvas.find_overlapping(x, y, x, y)):
            for tag in self.canvas.gettags(item):
                if tag in self.actions:
                    return tag
        return None

    def track_hover(self, event):
        if self.press is None:
            self.set_hover(self.hit_node(event.x, event.y))

    def set_hover(self, tag):
        self.canvas.configure(cursor="hand2" if tag is not None else "")
        if tag != self.hovered:
            self.hovered = tag
            if self.hover_timer is None:
                self.animate_hover()

    def animate_hover(self):
        self.hover_timer = None
        pending = False
        for tag in set(self.hover_levels) | ({self.hovered} if self.hovered is not None else set()):
            current = self.hover_levels.get(tag, 0)
            goal = int(tag == self.hovered)
            if current == goal:
                continue
            level = min(1, current + .2) if goal else max(0, current - .2)
            if abs(level - goal) < .001:
                level = goal
            item, fill, outline = self.hover_nodes[tag]
            self.canvas.itemconfigure(item, fill=self.blend(fill, "#cbdcff", level),
                                     outline=self.blend(outline, "#4d76bb", level),
                                     width=max(1, (1 + .7 * level) * self.scale))
            if level:
                self.hover_levels[tag] = level
            else:
                self.hover_levels.pop(tag, None)
            pending |= level != goal
        if pending:
            self.hover_timer = self.root.after(17, self.animate_hover)

    @staticmethod
    def blend(start, end, amount):
        channels = [round(int(start[i:i + 2], 16) * (1 - amount) + int(end[i:i + 2], 16) * amount)
                    for i in (1, 3, 5)]
        return "#{:02x}{:02x}{:02x}".format(*channels)

    def wheel(self, event):
        if self.block_bounds:
            x, y = self.canvas.canvasx(event.x) / self.scale, self.canvas.canvasy(event.y) / self.scale
            left, top, right, bottom = self.block_bounds
            if left <= x <= right and top <= y <= bottom:
                if event.delta:
                    direction = -1 if event.delta > 0 else 1
                    self.step_block(direction * max(1, abs(event.delta) // 120))
                    self.track_hover(event)
                return "break"
        return self.zoom_wheel(event)

    def zoom_wheel(self, event):
        if event.delta:
            self.zoom(1.13 if event.delta > 0 else 1 / 1.13, (event.x, event.y))
            self.track_hover(event)
        return "break"

    def key_step(self, event, direction):
        if self.block_bounds and not isinstance(event.widget, ttk.Combobox):
            self.step_block(direction)

    def step_block(self, amount):
        if not self.blocks:
            return
        index = max(0, min(len(self.blocks) - 1, self.block_index + amount))
        if index == self.block_index:
            return
        self.block_index = index
        if self.current:
            self.current = self.blocks[index]
            self.render()
        else:
            self.render_graph()

    def reset_zoom(self):
        self.zoom(1 / self.scale)

    def zoom(self, ratio, center=None):
        x, y = center if center is not None else (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        anchor_x, anchor_y = self.canvas.canvasx(x) / self.scale, self.canvas.canvasy(y) / self.scale
        self.scale = min(2.2, max(.55, self.scale * ratio))
        self.render_graph()
        width, height = (value * self.scale for value in self.extent)
        self.canvas.xview_moveto((anchor_x * self.scale - x) / width)
        self.canvas.yview_moveto((anchor_y * self.scale - y) / height)

    def render(self):
        for widget in self.breadcrumbs.winfo_children():
            widget.destroy()
        self.back.configure(state="normal" if self.current else "disabled")
        trail = []
        key = self.current
        while key:
            trail.append(key)
            key = self.parent(key)
        ttk.Button(self.breadcrumbs, text="Model", command=lambda: self.open_module("")).pack(side="left")
        for key in reversed(trail):
            ttk.Button(self.breadcrumbs, text=module_label(key),
                       command=lambda name=key: self.open_module(name)).pack(side="left", padx=(5, 0))
        self.render_graph()

    def label(self, x, y, text, *, size=12, color=TEXT, bold=False, tags=(), anchor="nw", width=0):
        self.canvas.create_text(x * self.scale, y * self.scale, text=text, fill=color, anchor=anchor,
                                width=round(width * self.scale), tags=tags,
                                # Canvas positions are pixels. Tk's negative font size uses pixels too.
                                font=(FONT, -max(7, round(size * self.scale)), "bold" if bold else "normal"))

    def node(self, x, y, width, height, title, subtitle="", *, target=None, tone=None):
        tags = ()
        if target is not None:
            key = "node-" + str(len(self.actions))
            self.actions[key] = target
            tags = (key,)
        fill, outline = COLORS.get(tone, (NODE, EDGE))
        item = self.canvas.create_rectangle(*(value * self.scale for value in (x, y, x + width, y + height)),
                                            fill=fill, outline=outline, width=max(1, self.scale), tags=tags)
        if tags:
            self.hover_nodes[tags[0]] = (item, fill, outline)
        self.label(x + 12, y + 10, title, size=12, bold=True, width=width - 24, tags=tags)
        if subtitle:
            self.label(x + 12, y + height - 11, subtitle, size=10, color=MUTED,
                       anchor="sw", width=width - 24, tags=tags)

    def wire(self, *points, arrow=True, dash=None):
        self.canvas.create_line(*(value * self.scale for value in points),
                                fill=WIRE, width=max(1, 1.4 * self.scale), dash=dash,
                                arrow="last" if arrow else "none",
                                arrowshape=tuple(max(3, round(value * self.scale)) for value in (8, 10, 4)))

    def frame(self, x, y, width, height, title):
        self.canvas.create_rectangle(*(v * self.scale for v in (x, y, x + width, y + height)),
                                     fill="#f8f6ff", outline="#9b87c7", width=max(1, 1.5 * self.scale))
        if title:
            self.label(x + 15, y + 13, title, size=11, bold=True, color="#624392")

    def block_switcher(self, x, y, width, *, open_block=False):
        block = self.blocks[self.block_index]
        self.node(x, y, 35, 30, "‹", target=-1 if self.block_index else None)
        self.node(x + 43, y, width - 86, 30, module_label(block),
                  target=block if open_block else None, tone="factor")
        self.node(x + width - 35, y, 35, 30, "›", target=1 if self.block_index + 1 < len(self.blocks) else None)

    def stage_stack(self, block, composition=False):
        """Ordered stage rows for one decoder block, derived from the adapter's flow + residual_after."""
        module = self.module_index.get(block) if block else None
        flow = (module or {}).get("flow") or []
        residual = (module or {}).get("residual_after") or []
        stages = []
        for child in flow:
            info = self.module_index.get(child)
            if not info:
                continue
            subtitle = self.composition_text(child) if composition else ""
            stages.append((module_label(child), KIND_TONES.get(info["kind"]), child, subtitle))
            word = RESIDUAL_WORDS.get(info["kind"])
            if word and child in residual:
                stages.append((f"Add {word} residual", "output", None, ""))
        return stages

    def composition_text(self, name):
        children = [self.module_index[child] for child in self.module_index[name].get("flow") or []
                    if child in self.module_index]
        projections = [info for info in children if info["kind"] in ("projection", "factorized")]
        if not projections:
            return ""
        ranks = sorted({info["rank"] for info in projections if info["kind"] == "factorized"})
        if all(info["kind"] == "factorized" for info in projections):
            return f"{len(projections)} factorized projections" + (f" · rank {ranks[0]}" if len(ranks) == 1 else "")
        return f"{len(projections)} projections" if ranks else "dense projections"

    def decoder_layers(self, x, y, width, height, gap, *, block=None, composition=False):
        stages = self.stage_stack(block, composition)
        middle = x + width / 2
        if not stages:
            self.node(x, y - 5 * gap, width, height + 5 * gap, "No decoder stages detected",
                      "This checkpoint does not expose a block dataflow")
            return
        residual_stages = [i for i, (_, tone, _, _) in enumerate(stages) if tone == "output"]
        for i, (title, tone, name, subtitle) in enumerate(stages):
            top = y - i * gap
            if i:
                self.wire(middle, top + gap, middle, top + height + (18 if subtitle else 0))
            self.node(x, top, width, height + (18 if subtitle else 0), title, subtitle, target=name, tone=tone)
        for j, index in enumerate(residual_stages):
            add = y - index * gap + height / 2
            start = y + height + 5 if not j else y - residual_stages[j - 1] * gap - 5
            self.wire(middle, start, x - 44 - 26 * j, start, x - 44 - 26 * j, add, x, add)

    def final_norm_name(self):
        if "model.norm" in self.module_index:
            return "model.norm"
        return next((name for name, module in self.module_index.items() if module["kind"] == "normalization"
                     and not any(name.startswith(block + ".") for block in self.blocks)), None)

    def embedding_name(self):
        if "model.embed_tokens" in self.module_index:
            return "model.embed_tokens"
        return next((name for name, module in self.module_index.items()
                     if any(len(tensor["shape"]) == 2 and "embed" in tensor["name"]
                            for tensor in module["tensors"])), None)

    def overview(self):
        self.label(35, 24, "TOKEN-TO-LOGIT FLOW", size=11, color=TEAL, bold=True)
        self.label(35, 45, "Read from input tokens at the bottom to output logits at the top.", color=MUTED)
        self.frame(82, 201, 415, 297, "")
        active = self.blocks[self.block_index] if self.blocks else None
        self.decoder_layers(174, 443, 270, 28, 39, block=active)
        for start, end in ((584, 565), (529, 471), (248, 190), (154, 139), (103, 91)):
            self.wire(309, start, 309, end)
        if active:
            self.block_bounds = (82, 201, 497, 498)
            # Keep the block control beside the output connection.
            self.block_switcher(96, 211, 193, open_block=True)
        self.node(174, 63, 270, 28, "Next-token logits", tone="output")
        head = "lm_head" if "lm_head" in self.module_index else "model.embed_tokens"
        self.node(174, 103, 270, 36, "Vocabulary head · tied weights" if head == "model.embed_tokens" else
                  "Vocabulary head", target=head if head in self.module_index else None, tone="embedding")
        self.node(174, 154, 270, 36, "Final RMSNorm", target=self.final_norm_name(), tone="norm")
        self.node(174, 529, 270, 36, "Token embedding", target=self.embedding_name(), tone="embedding")
        self.node(174, 584, 270, 36, "Input token IDs")
        self.label(550, 23, self.version["name"], size=24, bold=True)
        self.label(551, 59, self.version["representation"], color=MUTED, width=470)
        self.node(550, 97, 197, 63, size_text(self.version["bytes"]), "Checkpoint storage")
        self.node(763, 97, 197, 63, f"{self.version['parameters']:,}", "Stored parameters")
        self.label(550, 197, f"DECODER STACK · {len(self.blocks)} BLOCKS", size=11, color=TEAL, bold=True)
        self.label(550, 222, "Scroll inside the decoder frame or use its arrows to move through the stack.",
                   width=350, color=MUTED)
        self.label(550, 269, "Click the block label to open the full block, or click a colored stage to inspect it directly.",
                   width=350, color=MUTED)
        self.extent = (1030, 640)

    def block_view(self, module):
        stages = self.stage_stack(module["name"], True)
        last_top = 481 - (max(len(stages), 6) - 1) * 61
        flow = module.get("flow") or []
        prenorm = bool(flow) and self.module_index.get(flow[0], {}).get("kind") == "normalization"
        self.frame(82, 96, 490, 479, "PRE-NORMALIZED DECODER BLOCK" if prenorm else "DECODER BLOCK")
        self.block_bounds = (82, 96, 572, 575)
        self.decoder_layers(181, 481, 320, 32, 61, block=module["name"], composition=True)
        self.wire(341, 529, 341, 513)
        self.wire(341, last_top, 341, last_top - 16)
        self.node(181, 529, 320, 34, "Input hidden state")
        self.node(181, last_top - 51, 320, 35, "Output hidden state", tone="output")
        self.label(618, 110, "RESIDUAL STREAM", size=11, color=TEAL, bold=True)
        self.label(618, 135, "Each sublayer reads a normalized copy of the hidden state. "
                   "The bypass carries the original state to the addition node.", width=295, color=MUTED)
        self.node(618, 207, 295, 61, f"{module['output_width']:,} features", "Hidden width at the block boundary")
        self.block_switcher(618, 297, 295)
        self.extent = (1000, 613)

    def factor_extent(self, name, height):
        return height * 2 + 18 if self.module_index[name]["kind"] == "factorized" else height

    def flow_node(self, x, y, width, height, name, *, tone, title=None):
        """One flow child: a dense projection as a single box, a factorized one as chained B/A factor boxes."""
        module = self.module_index[name]
        if module["kind"] == "factorized":
            tensors = {tensor["name"]: tensor for tensor in module["tensors"]}
            right, left = tensors[module["matrix"]["right"]], tensors[module["matrix"]["left"]]
            self.node(x, y, width, height, "B · Input factor", shape_text(right["shape"]) + " · " + right["dtype"],
                      target=name, tone=tone)
            self.wire(x + width / 2, y + height, x + width / 2, y + height + 18)
            self.node(x, y + height + 18, width, height, "A · Output factor",
                      shape_text(left["shape"]) + " · " + left["dtype"], target=name, tone=tone)
            return y + height * 2 + 18
        subtitle = shape_text(module["shape"]) if module.get("shape") else ""
        self.node(x, y, width, height, title or module_label(name), subtitle, target=name, tone=tone)
        return y + height

    def attention_view(self, name):
        module = self.module_index[name]
        flow = module.get("flow") or []
        rope = any(key.startswith("rope") for key in self.config)
        q = self.config.get("num_attention_heads", "—")
        kv = self.config.get("num_key_value_heads", "—")
        head_dim = self.config.get("head_dim")
        if head_dim is None and isinstance(q, int) and isinstance(self.config.get("hidden_size"), int):
            head_dim = self.config["hidden_size"] // q
        grouped = "GROUPED-QUERY " if isinstance(q, int) and isinstance(kv, int) and kv < q else ""
        projections = [child for child in flow
                       if self.module_index.get(child, {}).get("kind") in ("projection", "factorized")]
        columns, tail = projections[:3], (projections[3] if len(projections) > 3 else None)
        follow = {}
        last = None
        for child in flow:
            kind = self.module_index.get(child, {}).get("kind")
            if kind in ("projection", "factorized"):
                last = child
            elif kind == "normalization" and last:
                follow.setdefault(last, []).append(child)
        bottoms, letters = [], []
        for child in columns:
            letter = child.rsplit(".", 1)[-1][0].upper()
            letters.append(letter)
            bottom = 188 + self.factor_extent(child, 55) + 50 * len(follow.get(child, []))
            if rope and letter in ("Q", "K"):
                bottom += 50
            bottoms.append(bottom)
        center = 453
        op_y = max(bottoms, default=243) + 46
        output_y = op_y + 89 + (self.factor_extent(tail, 54) if tail else 0) + 18
        self.frame(35, 99, 910, output_y + 32 + 19 - 99, f"CAUSAL {grouped}ATTENTION")
        self.node(343, 130, 220, 34, "Normalized hidden state")
        elbow = op_y - 21
        for i, child in enumerate(columns):
            x = (76, 343, 610)[i]
            mid = x + 110
            self.wire(center, 164, center, 175, mid, 175, mid, 188)
            bottom = self.flow_node(x, 188, 220, 55, child, tone="attention")
            for norm in follow.get(child, []):
                self.wire(mid, bottom, mid, bottom + 17)
                self.node(x, bottom + 17, 220, 33, f"{norm.rsplit('.', 1)[-1][0].upper()} RMSNorm",
                          target=norm, tone="norm")
                bottom += 50
            if rope and letters[i] in ("Q", "K"):
                self.wire(mid, bottom, mid, bottom + 17)
                self.node(x, bottom + 17, 220, 33, f"Rotary position embedding · {letters[i]}", tone="factor")
                bottom += 50
            if i == 1:
                self.wire(mid, bottom, mid, op_y)
            elif i:
                self.wire(mid, bottom, mid, elbow, center + 93, elbow, center + 93, op_y)
            else:
                self.wire(mid, bottom, mid, elbow, center - 93, elbow, center - 93, op_y)
        self.node(283, op_y, 340, 62, "Scaled dot-product attention",
                  f"{q} query heads · {kv} key/value heads · {head_dim} features/head", tone="attention")
        self.node(672, op_y + 10, 222, 55, "Causal mask", "Blocks attention to later tokens", tone="norm")
        self.wire(672, op_y + 34, 623, op_y + 34, dash=(4, 3))
        if tail:
            self.wire(center, op_y + 62, center, op_y + 89)
            bottom = self.flow_node(343, op_y + 89, 220, 54, tail, tone="attention")
        else:
            bottom = op_y + 62
        self.wire(center, bottom, center, output_y)
        self.node(343, output_y, 220, 32, "Attention output", tone="output")
        self.extent = (1000, output_y + 74)

    def mlp_view(self, name):
        module = self.module_index[name]
        flow = [child for child in module.get("flow") or [] if child in self.module_index]
        suffixes = {child.rsplit(".", 1)[-1] for child in flow}
        if {"gate_proj", "up_proj", "down_proj"} <= suffixes:
            act = str(self.config.get("hidden_act") or "").lower()
            self.gated_mlp_view(flow, act)
            return
        self.chain_view(module, input_title="Normalized hidden state", output_title="MLP output", tone="mlp")

    def gated_mlp_view(self, flow, act):
        children = {child.rsplit(".", 1)[-1]: child for child in flow}
        act_display = "SiLU" if act == "silu" else act.title()
        act_label = "SiLU activation" if act == "silu" else f"Activation ({act.title()})" if act else "Activation"
        gate_bottom = 190 + self.factor_extent(children["gate_proj"], 59)
        silu_y = gate_bottom + 22
        join_y = max(silu_y + 34, 190 + self.factor_extent(children["up_proj"], 59)) + 20
        multiply_y = join_y + 26
        down_y = multiply_y + 76
        output_y = down_y + self.factor_extent(children["down_proj"], 58) + 26
        variant = "SwiGLU" if act == "silu" else act.title()
        self.frame(35, 99, 805, output_y + 67 - 99, f"GATED FEED-FORWARD NETWORK · {variant}")
        self.node(325, 130, 225, 34, "Normalized hidden state")
        self.wire(438, 164, 438, 175, 229, 175, 229, 190)
        self.wire(438, 164, 438, 175, 644, 175, 644, 190)
        gate_bottom = self.flow_node(116, 190, 225, 59, children["gate_proj"], tone="mlp")
        up_bottom = self.flow_node(532, 190, 225, 59, children["up_proj"], tone="mlp")
        self.wire(229, gate_bottom, 229, silu_y)
        self.node(116, silu_y, 225, 34, act_label, tone="norm")
        self.wire(229, silu_y + 34, 229, join_y, 400, join_y, 400, multiply_y)
        self.wire(644, up_bottom, 644, join_y, 479, join_y, 479, multiply_y)
        self.node(325, multiply_y, 225, 53, "Elementwise multiplication", f"{act_display}(gate) × up", tone="factor")
        self.wire(438, multiply_y + 53, 438, down_y)
        down_bottom = self.flow_node(325, down_y, 225, 58, children["down_proj"], tone="mlp")
        self.wire(438, down_bottom, 438, output_y)
        self.node(325, output_y, 225, 34, "MLP output", tone="output")
        factorized = [child for child in flow if self.module_index[child]["kind"] == "factorized"]
        if factorized:
            ranks = " / ".join(map(str, sorted({self.module_index[child]["rank"] for child in factorized})))
            self.label(35, output_y + 12, f"{len(factorized)} factorized projections · rank {ranks}",
                       size=11, color=MUTED)
        self.extent = (1030, output_y + 99)

    def chain_view(self, module, *, input_title, output_title, tone=None):
        flow = [child for child in module.get("flow") or [] if child in self.module_index]
        center, width = 438, 225
        x = center - width // 2
        y = 164
        positions = []
        for child in flow:
            kind = self.module_index[child]["kind"]
            height = 58 if kind in ("projection", "factorized") else 34
            positions.append((child, y + 22, height,
                              tone if kind in ("projection", "factorized") else KIND_TONES.get(kind)))
            y += 22 + self.factor_extent(child, height)
        self.frame(35, 99, 805, y + 67 - 99, module_label(module["name"]).upper())
        self.node(x, 130, width, 34, input_title)
        bottom = 164
        for child, top, height, child_tone in positions:
            self.wire(center, bottom, center, top)
            bottom = self.flow_node(x, top, width, height, child, tone=child_tone)
        self.wire(center, bottom, center, bottom + 22)
        self.node(x, bottom + 22, width, 34, output_title, tone="output")
        self.extent = (1030, bottom + 121)

    def module_view(self, module):
        name = module["name"]
        self.label(35, 28, module_label(name), size=23, bold=True)
        self.label(35, 73, name, size=11, color=MUTED)
        kind = module["kind"]
        if kind == "transformer block":
            self.block_view(module)
            return
        if module.get("flow"):
            if kind == "self-attention":
                self.attention_view(name)
            elif kind == "feed-forward network":
                self.mlp_view(name)
            else:
                self.chain_view(module, input_title="Input", output_title="Output")
            return
        params = module.get("parameters", 0)
        if params:
            self.node(35, 112, 200, 64, f"{params:,}", "Parameters stored here")
            self.node(249, 112, 200, 64, size_text(module["bytes"]), "Weight payload stored here")
        elif module.get("output_width"):
            self.node(35, 112, 200, 64, f"{module['output_width']:,} features", "Block output width")
        children = self.children.get(name, [])
        if children:
            self.label(35, 210, "MODULE CONTAINMENT", size=9, color=TEAL, bold=True)
            self.node(35, 245, 230, 65, module_label(name), module["kind"], tone="factor")
            self.canvas.create_line(*(value * self.scale for value in (65, 310, 65, 357)),
                                    fill=EDGE, dash=(4, 3))
            for i, child in enumerate(children):
                info = self.module_index[child]
                subtitle = (shape_text(info["shape"]) if info.get("shape") else info["kind"])
                self.node(35 + (i % 3) * 265, 359 + (i // 3) * 94, 244, 74,
                          module_label(child), subtitle, target=child)
            self.extent = (880, max(600, 382 + ((len(children) + 2) // 3) * 94))
        elif module.get("matrix"):
            self.projection(module)
        else:
            self.label(35, 210, "STORED TENSORS", size=9, color=TEAL, bold=True)
            for index, tensor in enumerate(module["tensors"]):
                self.node(35, 245 + 80 * index, 360, 63, tensor["name"].rsplit(".", 1)[-1],
                          shape_text(tensor["shape"]) + " · " + tensor["dtype"])
            self.extent = (880, 580)

    def projection(self, module):
        out_width, in_width = module["shape"]
        tensors = {tensor["name"]: tensor for tensor in module["tensors"]}
        self.label(35, 211, "PROJECTION DATAFLOW", size=9, color=TEAL, bold=True)
        stages = [("Input x", f"{in_width:,} features", False)]
        if module["kind"] == "factorized":
            for key, label in (("right", "B · Input factor"), ("left", "A · Output factor")):
                tensor = tensors[module["matrix"][key]]
                stages.append((label, shape_text(tensor["shape"]) + " · " + tensor["dtype"], True))
            self.label(35, 361, f"y = A(Bx)   ·   rank {module['rank']}   ·   logical W {shape_text(module['shape'])}",
                       size=13, color=MUTED)
        else:
            tensor = tensors[module["matrix"]["dense"]]
            stages.append(("W · Dense weight", shape_text(tensor["shape"]) + " · " + tensor["dtype"], True))
            self.label(35, 361, "y = Wx", size=13, color=MUTED)
        stages.append(("Output y", f"{out_width:,} features", False))
        for i, (title, subtitle, accent) in enumerate(stages):
            x = 35 + 204 * i
            if i:
                self.wire(x - 31, 282, x - 4, 282)
            self.node(x, 244, 170, 76, title, subtitle, tone="factor" if accent else None)
        self.node(35, 406, 231, 50, "Inspect matrix", "Heatmap, norms and spectrum",
                  target=lambda: self.open_matrix(module["name"]), tone="output")
        self.extent = (880, 550)

    def open_matrix(self, name):
        from gptcompress.inspector.desktop_matrix import MatrixWindow

        if self.matrix_window and self.matrix_window.exists():
            self.matrix_window.open(self.version, name)
        else:
            self.matrix_window = MatrixWindow(self.root, self.versions, self.version, name)

    def open_comparison(self):
        from gptcompress.inspector.desktop_comparison import ComparisonWindow

        if self.comparison_window and self.comparison_window.exists():
            self.comparison_window.window.lift()
            self.comparison_window.window.focus_set()
        else:
            self.comparison_window = ComparisonWindow(self.root, self.comparison_versions, self.version["id"])

    def render_graph(self):
        if self.hover_timer is not None:
            self.root.after_cancel(self.hover_timer)
            self.hover_timer = None
        self.hovered = None
        self.hover_levels.clear()
        self.hover_nodes.clear()
        self.block_bounds = None
        self.canvas.configure(cursor="")
        self.canvas.delete("all")
        self.actions.clear()
        if self.current:
            self.module_view(self.module_index[self.current])
        else:
            self.overview()
        self.canvas.configure(scrollregion=(0, 0, *(value * self.scale for value in self.extent)))
        self.zoom_label.configure(text=f"{round(100 * self.scale)}%")


def find_workspace(explicit):
    if explicit:
        return explicit.resolve()
    if not getattr(sys, "frozen", False):
        return inspection.WORK
    for directory in Path(sys.executable).resolve().parents:
        if (directory / "WORKER_REQUIREMENTS.md").is_file() and (directory / "models").is_dir():
            return directory
    raise ValueError("Place the executable inside the GptCompress Work folder or its dist folder.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    root = tk.Tk()
    root.withdraw()
    try:
        inspection.WORK = find_workspace(args.workspace)
        inspection.STORE = inspection.WORK / "inspection"
        inspection.discover()
        registered = inspection.versions()
        if not registered:
            raise ValueError("No registered checkpoints were found. Check the Work/models directory.")
        ArchitectureWindow(root, registered, registered)
    except (OSError, ValueError, KeyError) as error:
        messagebox.showerror("GptCompress Inspector", str(error), parent=root)
        root.destroy()
        return
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
