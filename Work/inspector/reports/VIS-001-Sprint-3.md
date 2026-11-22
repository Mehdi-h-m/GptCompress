# Desktop Inspector · Sprint 3

Double-click `Work/inspector/dist/model-inspector/versions/GptCompress-Model-Inspector-Sprint-3.exe` and choose **Compare models**. The dashboard loads saved experiment records and draws paired bar charts.

## Review walkthrough

1. Choose **Qwen3-4B** as A and **als-r256-mlp** as B. The storage chart shows **7.51 GiB → 3.13 GiB**, a 58.3% reduction relative to A.
2. Check the matched development scores. The run used ten questions per task for each version. GSM8K flexible exact match is **80% → 0%**. IFEval strict prompt accuracy is **60% → 10%**.
3. Click **Swap A/B**. The score differences become **+80** and **+50 percentage points**. Storage and matched resource percentage changes are recalculated using the new Version A as the reference.
4. Select the saved hybrid smoke run for each model. The VRAM chart displays the recorded peaks. The run note shows the number of generated tokens in each run, and the charts display the separate measurements.

The default matched CPU development runs measured **2h 54m 53s → 1h 42m 15s** in elapsed wall time and **8.34 GiB → 5.08 GiB** in peak process working set. Peak VRAM was unmeasured on those runs.

## Data and matching

Benchmark deltas require completed scores with the same metric, protocol hash, exact ordered question IDs, and sample counts. The protocol selector switches between matched benchmark protocols when several exist. Resource deltas require complete paired benchmark workloads with the same protocol, ordered question IDs, device, and CPU thread count. Independently selected runs can still show their recorded absolute resource values. Missing resource values appear as **Not measured**.

Registry IDs and run IDs are saved locally in `inspection/desktop_comparison.json`. The dashboard reads existing artifacts from `experiments/`. It uses the recorded elapsed wall time, peak process working set and peak CUDA memory reserved by PyTorch. No new benchmark execution is needed.

## Historical verification and build record

The native UI tests exercise the real Qwen comparison, switching resource runs, swapping scores and runs, restoring the saved pair, and selecting the same model twice. Focused data checks exercise mismatched question IDs, differences in execution settings, missing measurements and measured zero. All 26 tests passed. A smoke test launched the packaged executable from a different working directory, clicked **Compare models**, and verified the rendered dashboard against the saved records.

The commands below are the exact Sprint 3 commands previously run, preserved for provenance. Their source and output paths describe the old layout; they do not rebuild this historical binary from the current source. The current entry point and commands are in the [runbook](../README.md#desktop-model-inspector) and [Sprint 4 build instructions](VIS-001-Sprint-4.md#verification-and-rebuild).

```powershell
rtk proxy .venv/Scripts/python.exe -m unittest discover -s inspector/tests -v
rtk proxy uv venv .cache/inspector-build-env --python .venv/Scripts/python.exe
rtk proxy uv pip install --python .cache/inspector-build-env/Scripts/python.exe pyinstaller==6.22.3 numpy==2.5.5
rtk proxy .cache/inspector-build-env/Scripts/python.exe -m PyInstaller --noconfirm --onefile --windowed --name GptCompress-Inspector-Sprint3 --distpath dist --workpath .cache/inspector-build --specpath .cache/inspector-build --exclude-module safetensors --exclude-module torch --exclude-module transformers --exclude-module accelerate --exclude-module psutil src/inspect_desktop.py
```
