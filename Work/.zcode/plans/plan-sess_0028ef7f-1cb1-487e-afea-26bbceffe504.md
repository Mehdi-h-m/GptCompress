# Sprint 4 — Data-driven architecture renderer

## Context

The desktop inspector (`src/inspect_desktop.py`) draws hardcoded dense-Qwen3 diagrams: a fixed 6-stage block flow, fixed attention/MLP layouts, hardcoded tensor names, and only `down_proj` gets a "· rank N" suffix. So `als-r256-mlp` (108 factorized MLP projections, rank 256) renders identically to `Qwen3-4B`. The registry data is already correct (`inspection/versions.json` records `kind: "factorized"`, rank, A/B tensors, 1.67B vs 4.02B params) — the renderer just ignores it. This refactor makes every diagram derive from each checkpoint's actual module tree, rendering factorized projections as expanded B→A factor pairs (user-selected). Delivered as "Sprint 4" per project convention (report + rebuilt exe).

Per `AGENTS.md` roles: the master (me) writes this assignment; **a worker agent implements it**. The worker must read `WORKER_REQUIREMENTS.md` before starting (VIS-001 item 5 already mandates: "Display factorized projection structure and its logical relationship to W"; "Neither is hardcoded to the original model"). Workers must preserve unrelated user changes — note `README.md` and `WORKER_REQUIREMENTS.md` already have uncommitted user edits; README changes must be additive only.

## 1. Data layer — `src/gptcompress/inspection.py`

Move family dataflow knowledge out of the renderer into the family adapter (VIS-001 item 2: distinguish hierarchy from verified dataflow edges):

- Add a helper (e.g. `_attach_flow(modules)`) called at the **end of `components()` in each adapter** — it must run *after* factor merging, because in a factorized checkpoint the base pass sees only `*.left`/`*.right` leaf modules, not the merged projections.
- `Qwen3Adapter`: for each container, attach `flow` (ordered child names, filtered to children that actually exist in `modules`):
  - block: `input_layernorm, self_attn, post_attention_layernorm, mlp`; plus `residual_after = ["self_attn", "mlp"]` only when the pre-norm pattern holds (first flow entry is a norm).
  - `self_attn`: `q_proj, q_norm, k_proj, k_norm, v_proj, o_proj`.
  - `mlp`: `gate_proj, up_proj, down_proj`.
- `FactorizedQwen3Adapter`: call the same helper after merging `.left`/`.right` into `kind: "factorized"` modules (child names resolve unchanged).
- Generic `CheckpointAdapter`: no flow (leaf-only tree) — renderer falls back to the existing containment grid.

## 2. Renderer — `src/inspect_desktop.py`

Replace all fixed structure with data-driven rendering. **Constraint: keep pixel parity for the 6-stage qwen3 flow** (same constants: overview 28h/39gap, block view 32h/61gap, same `block_bounds` frames) so existing coordinate-based tests pass unchanged; dynamic geometry only kicks in for other flows.

- `decoder_layers()` → generic stage-stack builder driven by `module["flow"]` + `residual_after`: stage nodes titled via `module_label(child)` (already yields "Input RMSNorm", "Self-attention"-routed views, etc.), tones from child `kind`; residual add nodes only for `residual_after` children. Handle zero-block checkpoints (placeholder instead of `IndexError`).
- `attention_view()` → driven by the container's `flow`:
  - New projection-node helper: dense → one box with `shape_text`; **factorized → two chained boxes "B · Input factor" (rank×in) and "A · Output factor" (out×rank)**, shapes/dtype read from the module's actual factor tensors (reuse existing label vocabulary from `projection()`).
  - Norm children after their projection; Q/K RoPE nodes rendered only when `config.json` has `rope*` keys; attention-op node subtitle from config (`num_attention_heads`/`num_key_value_heads`/`head_dim` with fallback `hidden_size // num_attention_heads`, keep current subtitle wording); causal-mask node stays (decoder-only semantics).
- `mlp_view()` → driven by `flow`:
  - gate+up+down present → branch layout: per-branch projection rendered factor-aware, activation node labeled from `config["hidden_act"]` (silu → "SiLU activation", fallback "Activation ({value})"), elementwise-mult node with subtitle `f"{act}(gate) × up"` shown only when the gated pattern is present.
  - Otherwise → generic sequential chain of children (each factor-aware).
- Container nodes in `block_view`/stage stacks get a derived subtitle summarizing composition, e.g. "3 factorized projections · rank 256" or "dense projections".
- `block_view()` frame title derived ("PRE-NORMALIZED DECODER BLOCK" when the pre-norm pattern holds, else "DECODER BLOCK").
- `overview()`: vocabulary head = `lm_head` if present else tied `model.embed_tokens` (existence-checked); final norm and token embedding resolved by known names with data-derived fallback (norm-kind module outside blocks; module containing a 2-D `*embed*` tensor); stage stack from the active block's flow; keep storage/params/representation panel (already data-driven).
- `module_view()` routing: block → block_view; container with flow → attention/branch/chain view by kind; leaf with matrix → `projection()` (unchanged, already dynamic); leaf without matrix → tensors list (unchanged).
- `open_module()`: replace hardcoded `name.split(".")[:3]` with block depth derived from `self.blocks[0]` component count.
- `main()`: drop the `family == "qwen3" and "capture"` filter — select from **all registered versions** (the window needs only inventory+matrix); update the empty-registry error message.
- `LABELS` stays as a presentation-only lookup (never assumed present).

## 3. Tests — `tests/test_desktop_inspector.py`

- Preserve helper contracts the tests rely on: `app.actions` (tag→str/int/callable), `hover_nodes` tag→(item, fill, outline), `blocks`, `block_bounds`, `current`, `block_index`, `zoom_label`, `labels()` text-subset assertions, the "Compare models" button, and `click_node` by target value.
- Existing label assertions keep passing (qwen3-derived wording is preserved: `module_label` of flow children yields identical strings; "SiLU activation"/RoPE/head-subtitle derived from the same config values).
- Add tests:
  1. Factorized MLP view (als-r256-mlp, `model.layers.0.mlp`): labels contain factor boxes with real shapes ("256 × 2,560 · BF16", "9,728 × 256 · BF16" for gate; "256 × 9,728 · BF16", "2,560 × 256 · BF16" for down) plus rank mention; dense model's MLP view shows no factor boxes.
  2. Overview shows differing storage/parameters between the two versions.
  3. Generic fallback: synthesize a tiny checkpoint in a temp dir (config.json with a non-qwen3 `model_type`, untied `lm_head` + embedding, small `.safetensors` via `safetensors.torch.save_file`), describe it with `CheckpointAdapter`, render `ArchitectureWindow` — no crash, containment view renders, untied head and zero-block overview handled.
- Run: `rtk proxy .venv/Scripts/python.exe -m unittest discover -s tests -v` — all pass; record the count.

## 4. Deliverables per project convention

- `reports/VIS-001-Sprint-4.md` following the established sprint format (launch line, Review walkthrough showing the now-visible differences, Verification and rebuild with test command + PyInstaller command).
- `README.md`: additive sprint-4 note only.
- Rebuild `dist/GptCompress-Inspector-Sprint4.exe` **and** refresh the stale `dist/GptCompress-Inspector.exe` (built 13:28, predates six source edits) with the Sprint-3 recipe: pyinstaller 6.22.3 + numpy 2.5.5 in a scratch venv, `--exclude-module safetensors --exclude-module torch --exclude-module transformers --exclude-module accelerate --exclude-module psutil`, entry `src/inspect_desktop.py`.
- Launch the rebuilt exe as a smoke test (starts, no error dialog).

## Verification

1. Full suite green: `rtk proxy .venv/Scripts/python.exe -m unittest discover -s tests -v`.
2. Run the app from source against the two registered checkpoints: als-r256-mlp MLP views show B→A factor pairs; Qwen3-4B views unchanged; version switching preserves selection; block scrolling still works.
3. Master (me) reviews the worker's diff, re-runs the test suite, and spot-checks the rendered behavior before accepting.

## Non-goals

- No web app (`app.js`) changes; no matrix/comparison-window logic changes; no changes to compression/evaluation code; no unification of the three factor-label vocabularies (desktop keeps "B · Input factor"/"A · Output factor"); no new compression features (VIS-001 boundary).

## Files

- `src/gptcompress/inspection.py` — flow/residual attachment in adapters
- `src/inspect_desktop.py` — data-driven rendering of all view methods + version filter
- `tests/test_desktop_inspector.py` — new tests, minimal edits to existing
- `reports/VIS-001-Sprint-4.md` (new), `README.md` (additive), `dist/GptCompress-Inspector-Sprint4.exe` + `dist/GptCompress-Inspector.exe` (rebuilt)