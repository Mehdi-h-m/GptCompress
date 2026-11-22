# Desktop Inspector · Sprint 1

Open `Work/inspector/dist/model-inspector/versions/GptCompress-Model-Inspector-Sprint-1.exe` from File Explorer. The Windows build includes its Python and Tk runtime and reads the existing checkpoint inventory from the enclosing `Work` folder.

## Review walkthrough

```text
Qwen3-4B
  └─ Block 0
      ├─ Self-attention
      │   └─ Query, key, value and output projections
      └─ Feed-forward network
          └─ Down projection
              └─ x → W (2,560 × 9,728) → y

Switch to als-r256-mlp at the same projection
              └─ x → B (256 × 9,728) → A (2,560 × 256) → y
```

The white model diagram shows the embedding, all 36 decoder blocks, final RMSNorm and the tied vocabulary head. The block diagrams show the two pre-normalized sublayers and both residual bypasses. Attention expands into the Q, K and V projections, Q/K normalization, rotary position embeddings, causal grouped-query attention and the output projection. The feed-forward diagram expands into the gate and up branches, SiLU, multiplication and the down projection. The version selector preserves the selected module if it exists in both versions.

The overview presents one active block directly inside the decoder diagram. Scrolling over that frame, clicking its arrows, or pressing the left and right arrow keys changes the block. Clicking its label opens the full block. Clickable stages fade to a highlight on hover and restore their original colors when the pointer leaves.

The native UI checks exercise real checkpoint inventories, scrolling between blocks, hover transitions, diagram clicks, attention and MLP navigation, dense and factorized weight dimensions, version switching, zoom and drag-to-pan. The packaged window was launched from a different working directory to verify checkpoint discovery relative to the executable.

## Historical build record

The commands below are the exact Sprint 1 commands previously run, preserved for provenance. Their source and output paths describe the old layout; they do not rebuild this historical binary from the current source. The current entry point and commands are in the [runbook](../README.md#desktop-model-inspector) and [Sprint 4 build instructions](VIS-001-Sprint-4.md#verification-and-rebuild).

```powershell
rtk proxy uv venv .cache/inspector-build-env --python .venv/Scripts/python.exe
rtk proxy uv pip install --python .cache/inspector-build-env/Scripts/python.exe pyinstaller==6.22.3
rtk proxy .cache/inspector-build-env/Scripts/python.exe -m PyInstaller --noconfirm --onefile --windowed --name GptCompress-Inspector-Sprint1 --distpath dist --workpath .cache/inspector-build --specpath .cache/inspector-build --exclude-module numpy --exclude-module safetensors --exclude-module torch --exclude-module transformers --exclude-module accelerate --exclude-module psutil src/inspect_desktop.py
rtk proxy .venv/Scripts/python.exe -m unittest discover -s inspector/tests -v
```
