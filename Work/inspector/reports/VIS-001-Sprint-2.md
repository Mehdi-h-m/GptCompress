# Desktop Inspector · Sprint 2

Double-click `Work/inspector/dist/model-inspector/versions/GptCompress-Model-Inspector-Sprint-2.exe` to review the native matrix inspector. The bundled runtime reads the existing checkpoint files from the enclosing Work folder.

## Review walkthrough

Open **Block 0 → Gated feed-forward network → Down projection → Inspect matrix**.

- **Qwen3-4B** opens the dense BF16 weight with shape 2,560 × 9,728.
- **als-r256-mlp** opens the reconstructed logical matrix for the same slice. The weight menu also exposes the stored A and B factors with shapes 2,560 × 256 and 256 × 9,728.
- The mouse wheel and clicks zoom around a weight in the heatmap. Exact start coordinates and a slice-size selector allow inspection across the matrix. Hovering reports the underlying coordinate or coordinate range.
- The four charts show the weight histogram, row and column norms, and optional singular values. The summary includes minimum, maximum, mean, standard deviation, Frobenius norm and nonfinite count.

## Measurement method

Each request reads up to 512 × 512 values. BF16 weights are widened to FP32. The logical view computes the selected tile from the corresponding rows of A and columns of B in FP32. The heatmap uses at most 64 cells on each axis and displays the signed largest magnitude in each group of weights. At finer zoom each cell can show one weight. Statistics and histograms use all finite values in the slice. The optional SVD uses FP64 arithmetic on that slice. SVD is unavailable if any value in the slice is nonfinite.

Weights stay in local safetensors files. A worker reads each bounded slice and computes the charts outside the UI thread. A 16-entry cache is keyed by checkpoint identity, source tensor, file fingerprint, slice coordinates and spectrum setting. Matrix totals and matched model-level chart comparisons are separate sprint work.

## Historical verification and build record

The native UI tests cover opening the matrix window from the architecture diagram, rendering the heatmap, clicking to zoom, retaining the slice when switching models, selecting a factor, requesting singular values, and recovering from an invalid offset. Tensor tests independently check the dense and factor values against the safetensors reader and reconstruct a real ALS tile from both factors.

All 23 tests passed. The packaged executable opened a real Qwen3-4B projection and rendered its weight heatmap, histogram, and row and column norms. The native UI test also verified the logical ALS reconstruction, factor selection, and an on-demand 64-value spectrum.

The commands below are the exact Sprint 2 commands previously run, preserved for provenance. Their source and output paths describe the old layout; they do not rebuild this historical binary from the current source. The current entry point and commands are in the [runbook](../README.md#desktop-model-inspector) and [Sprint 4 build instructions](VIS-001-Sprint-4.md#verification-and-rebuild).

```powershell
rtk proxy .venv/Scripts/python.exe -m unittest discover -s inspector/tests -v
rtk proxy uv venv .cache/inspector-build-env --python .venv/Scripts/python.exe
rtk proxy uv pip install --python .cache/inspector-build-env/Scripts/python.exe pyinstaller==6.22.3 numpy==2.5.5
rtk proxy .cache/inspector-build-env/Scripts/python.exe -m PyInstaller --noconfirm --onefile --windowed --name GptCompress-Inspector-Sprint2 --distpath dist --workpath .cache/inspector-build --specpath .cache/inspector-build --exclude-module safetensors --exclude-module torch --exclude-module transformers --exclude-module accelerate --exclude-module psutil src/inspect_desktop.py
```
