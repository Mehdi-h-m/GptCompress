# Desktop Inspector · Sprint 4

Double-click `Work/inspector/dist/model-inspector/versions/GptCompress-Model-Inspector-Sprint-4.exe`. Every architecture diagram is built from the selected checkpoint's own module tree, so the compressed checkpoint looks different from the original everywhere, not only in the matrix window. The canonical latest build is `Work/inspector/dist/model-inspector/GptCompress-Model-Inspector.exe`.

## Review walkthrough

1. Select **Qwen3-4B** and note the overview panel: **7.51 GiB** checkpoint storage and **4,022,468,096** stored parameters. Switch to **als-r256-mlp**: **3.13 GiB** and **1,672,609,280**. The decoder stack keeps the same six stages in both, derived from each checkpoint's block flow.
2. With **als-r256-mlp** selected, open block 0 and click the **Feed-forward network** stage. Each compressed projection now renders as two chained factor boxes: **B · Input factor** (`256 × 2,560 · BF16`) above **A · Output factor** (`9,728 × 256 · BF16`) for the gate and up branches, and `256 × 9,728 · BF16` / `2,560 × 256 · BF16` for the down projection. A muted note records **3 factorized projections · rank 256**.
3. Switch the version picker back to **Qwen3-4B** while still inside the feed-forward network. The same view shows single dense boxes (**9,728 × 2,560** and **2,560 × 9,728**) with no factor boxes and no rank note.
4. Open the **Self-attention** stage in either version: the projections stay dense in both checkpoints, so the query/key/value columns, **Q/K RMSNorm** nodes, rotary embeddings, and the **32 query heads · 8 key/value heads · 128 features/head** panel are identical. Only the MLP differs, matching what the checkpoint actually stores.
5. Open **Down projection** from either feed-forward network for the per-projection dataflow view with **Inspect matrix**. It is unchanged from Sprint 2 and now agrees with the container views: factorized modules show **B · Input factor** then **A · Output factor** with the same shapes.

## Data and rendering

The family adapters attach dataflow metadata to the module tree during import: each `model.layers.{i}` block gets an ordered `flow` of existing children (`input_layernorm`, `self_attn`, `post_attention_layernorm`, `mlp`) plus a `residual_after` list when the pre-norm pattern holds; `self_attn` gets `q_proj → q_norm → k_proj → k_norm → v_proj → o_proj` and `mlp` gets `gate_proj → up_proj → down_proj`, each filtered to children that exist. The factorized adapter re-runs that attachment after merging `*.left`/`*.right` leaves into logical projections, so compressed MLP children are present in the flow. The generic `CheckpointAdapter` attaches no flow, and the renderer falls back to a placeholder stack and plain tensor listings for such checkpoints.

Stage titles, tones, click targets, residual additions, frame titles (`PRE-NORMALIZED DECODER BLOCK` is derived from the flow's first child being a normalization), the container stage subtitles (`3 factorized projections · rank 256`, `dense projections`), attention heads, RoPE presence (any `rope*` config key), the activation label (`SiLU` from `hidden_act`), and the pipeline anchors (vocabulary head, final norm, token embedding) are all resolved from version data at render time. Factorized projections render as a **B · Input factor** box wired into an **A · Output factor** box using the stored factor shapes and dtypes; dense projections render one box with the logical shape. `Qwen3-4B` and `als-r256-mlp` produce identical geometry for the dense views and the overview stage stack.

## Verification and rebuild

The historical native UI tests covered navigation over both real checkpoints, the factor-pair assertions on the compressed feed-forward network with the dense cross-check on **Qwen3-4B**, the storage/parameter difference in the overview, matrix inspection, hover, zoom, and the comparison dashboard. A separate test built a tiny synthetic non-Qwen3 checkpoint in a temporary directory and verified the generic adapter path renders an overview without blocks, keeps leaves clickable, and lists stored tensors. The recorded run passed all 29 tests. A smoke test launched the Sprint 4 executable, confirmed the process was alive after 12 seconds, and stopped it. These are historical results; the refactor did not rerun them.

```powershell
rtk proxy .venv/Scripts/python.exe -m unittest discover -s inspector/tests -v
rtk proxy uv venv .cache/inspector-build-env --python .venv/Scripts/python.exe
rtk proxy uv pip install --python .cache/inspector-build-env/Scripts/python.exe pyinstaller==6.22.3 numpy==2.5.5
rtk proxy .cache/inspector-build-env/Scripts/python.exe -m PyInstaller --noconfirm --onefile --windowed --name GptCompress-Model-Inspector-Sprint-4 --distpath dist/model-inspector/versions --workpath .cache/inspector-build --specpath .cache/inspector-build --paths src --exclude-module safetensors --exclude-module torch --exclude-module transformers --exclude-module accelerate --exclude-module psutil src/gptcompress/inspector/desktop_app.py
rtk proxy .cache/inspector-build-env/Scripts/python.exe -m PyInstaller --noconfirm --onefile --windowed --name GptCompress-Model-Inspector --distpath dist/model-inspector --workpath .cache/inspector-build --specpath .cache/inspector-build --paths src --exclude-module safetensors --exclude-module torch --exclude-module transformers --exclude-module accelerate --exclude-module psutil src/gptcompress/inspector/desktop_app.py
```
