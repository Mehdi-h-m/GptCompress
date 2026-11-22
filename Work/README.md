# GptCompress workspace

Three projects share this workspace:

| Folder | Project | Reports |
| --- | --- | --- |
| `gptcompress/` | Compression method: factorization solver, evaluation harness, experiment records | ALS-001, EVAL-001 in `gptcompress/reports/` |
| `inspector/` | Desktop model inspector app (VIS-001); its code is the `gptcompress.inspector` package | VIS-001 and Sprints 1–4 in `inspector/reports/` |
| `drm/` | Distributed Resource Manager — experiment orchestrator app (design stage) | `drm/docs/ORCHESTRATOR_ARCHITECTURE.md` |

Shared: [docs/DATA_ARCHITECTURE.md](docs/DATA_ARCHITECTURE.md) (data rules), [reports/index.html](reports/index.html) (generated dashboard), [WORKER_REQUIREMENTS.md](WORKER_REQUIREMENTS.md) (coordination). Dot-prefixed directories (`.venv`, `.cache`, `.hf-cache`, `.nltk-data`) are local runtime state and are not versioned.

Commands below run from `Work` in PowerShell.

## Desktop model inspector

Double-click [GptCompress-Model-Inspector.exe](inspector/dist/model-inspector/GptCompress-Model-Inspector.exe) in `Work/inspector/dist/model-inspector`. This is the canonical latest build. It finds checkpoints in the enclosing `Work` folder and opens a native Windows window. Historical sprint builds are kept in `inspector/dist/model-inspector/versions/`.

For Sprint 1, follow the decoder from input tokens to output logits. Scroll inside the decoder frame or click its arrows to change the active block. Click the block label to open the full block, or click an attention or feed-forward node to go straight to that stage. Hover highlights clickable nodes. Open the **Down projection** and switch to **als-r256-mlp** to see the same projection as two factors. Drag the canvas to pan, use **Ctrl+scroll** to zoom, and use the breadcrumbs to return to a parent module.

For Sprint 2, click **Inspect matrix** on a projection diagram. Scroll or click on the heatmap to zoom into a slice, or enter row and column coordinates. In the compressed checkpoint, the weight menu switches between the logical matrix and the individual factors. Enable **Singular values** to compute the spectrum for the visible slice.

For Sprint 3, click **Compare models** in the main window. Choose Version A and Version B independently. **Swap A/B** reverses the pair and its selected resource runs. The charts show checkpoint size, stored parameters, matched benchmark scores, and recorded elapsed time, RAM and VRAM. The app restores the pair and runs when you reopen the comparison.

For the Sprint 4 build, launch [GptCompress-Model-Inspector-Sprint-4.exe](inspector/dist/model-inspector/versions/GptCompress-Model-Inspector-Sprint-4.exe). The architecture diagrams are derived from each checkpoint's own module tree: in **als-r256-mlp**, the feed-forward network renders each compressed projection as chained **B · Input factor** and **A · Output factor** boxes with a rank 256 note, while **Qwen3-4B** shows dense weights. The overview reports **7.51 GiB / 4,022,468,096** versus **3.13 GiB / 1,672,609,280**. Attention stays dense in both checkpoints and renders identically.

## Source projects

- `gptcompress/src/gptcompress/compression/` contains the compression command and factorization code.
- `gptcompress/src/gptcompress/evaluation/` contains asset preparation, evaluation, experiment monitoring, and report generation.
- `gptcompress/src/gptcompress/inspector/` contains the web and desktop inspectors, inspection/recording/comparison code, and browser assets.
- Shared runtime helpers such as the CPU linear kernel and GPU placement remain in `gptcompress/src/gptcompress/`.

See [Sprint 3](inspector/reports/VIS-001-Sprint-3.md) for the comparison feature history and [Sprint 4](inspector/reports/VIS-001-Sprint-4.md) for current verification and rebuild instructions. [Sprint 2](inspector/reports/VIS-001-Sprint-2.md) covers matrix inspection, [Sprint 1](inspector/reports/VIS-001-Sprint-1.md) covers architecture navigation, and [VIS-001](inspector/reports/VIS-001.md) records earlier experiments and checkpoint metadata.

## Progress monitor

Open the read-only Python desktop monitor from `Work`:

```powershell
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/experiment_monitor.py
```

The window refreshes every five seconds. It shows completed matrices, benchmark counts when the harness logs them, and task statuses. Select a task for details, or add `--snapshot` for a terminal view.

## Fixed-rank ALS experiments

The weight-only solver alternates rank-revealing CPU least-squares solves in FP32 and stores BF16 factors. Each converted projection executes two linear operations, with a BF16 intermediate when evaluated in BF16. Reconstruction records include error before and after factor storage conversion. Rank is fixed within each run. Matrices with no parameter savings are recorded as skipped.

```powershell
rtk proxy .venv/Scripts/python.exe -m unittest discover -s gptcompress/tests -p test_factorization.py
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/compression/compress_model.py --rank 256 --probe --output gptcompress/experiments/ALS-001-probe-r256
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/compression/compress_model.py --rank 256 --target mlp --output gptcompress/models/compressed/als-r256-mlp
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/compression/compress_model.py --rank 256 --target attention --output gptcompress/models/compressed/als-r256-attention
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/compression/compress_model.py --rank 256 --target combined --reuse-factors gptcompress/models/compressed/als-r256-mlp gptcompress/models/compressed/als-r256-attention --output gptcompress/models/compressed/als-r256-combined
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --model gptcompress/models/compressed/als-r256-mlp --smoke --linear-kernel fp32-tiled
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --profile dev --batch-size 2 --linear-kernel fp32-tiled
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --model gptcompress/models/compressed/als-r256-mlp --profile dev --batch-size 2 --linear-kernel fp32-tiled
```

Use a new output directory for every conversion attempt. A failed conversion retains its progress record and cannot be evaluated as a complete asset. Complete variants are standalone, and the local evaluator selects their factorized loader automatically. Factor reuse checks hashes and solver settings. External runtimes need equivalent factorized-module support. Exact SVD comparisons are performed only in probe mode, which samples MLP down and attention key projections from blocks 0, 18, and 35.

The existing development and held-out profiles provide matching comparisons. Choose the final candidate using development scores and size, then compare it and the original using `--profile heldout` with identical batch/kernel settings. Accuracy trade-offs and CPU inference timing must be interpreted separately from weight reconstruction error.

Run these commands in `Work` with PowerShell. The workspace keeps full checkpoints in `gptcompress/models/original`, variants in `gptcompress/models/compressed`, immutable benchmark snapshots in `gptcompress/datasets/benchmarks`, and records in `gptcompress/experiments`. Keep compression calibration inputs in `gptcompress/datasets/calibration` and source them independently of the benchmarks.

```powershell
rtk proxy uv pip install --no-cache --python .venv/Scripts/python.exe --index https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
rtk proxy uv pip install --no-cache --python .venv/Scripts/python.exe -r requirements.txt
rtk proxy .venv/Scripts/python.exe -m nltk.downloader -d .nltk-data punkt_tab
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/prepare_assets.py model
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/prepare_assets.py data
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/prepare_assets.py evalplus
```

The asset commands use public endpoints and pin each dataset to the revision recorded on its first fetch. They resume partial downloads. Check each manifest for available files, hashes, licenses supplied by the dataset card, and access errors. The all-datasets command exits nonzero if a source is blocked and still attempts the other sources.

```powershell
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --smoke
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --profile pilot
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --profile dev
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --profile heldout
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --report
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --validate-profiles
```

The runnable profiles cover GSM8K and IFEval. Development uses 10 fixed rows per task. Held-out evaluation uses the next 100 rows from the same seeded deterministic permutation. The ordered row IDs and dataset hashes are recorded in the run manifest. GSM8K retains the harness's five-shot protocol using training rows and its published answer filters. The headline score uses flexible extraction. IFEval uses its published strict prompt-level scoring and records all four scores. The profiles use Qwen chat formatting with thinking disabled, greedy decoding, batch size one, and 2,048 total tokens. Output limits are 192 tokens for GSM8K and 256 for IFEval. These are custom subset results under bounded decoding. Treat the reported output-length flag as a diagnostic based on re-tokenized text.

The local pilot uses the same 10 development IDs per task and scores with the same harness. It uses a batch of ten and 512 total tokens. GSM8K uses a zero-shot prompt asking for the final number and allows 24 output tokens. IFEval preserves the input instructions and allows 48 output tokens. The pilot measures the end-to-end workflow and carries its own comparison protocol. The disjoint held-out profile uses the larger budgets above.

The original model runs on CPU with its BF16 checkpoint and safetensors loading. Windows reports peak process working set. The report shows VRAM as unavailable for the CPU run. Elapsed times include generation and harness overhead, and the run records model load time in its overall wall time.

`--linear-kernel fp32-tiled` speeds up BF16 projections on the tested AVX2 CPU. The forward pass casts temporary weight tiles of up to 64 MiB and inputs to FP32 for the matrix product, then rounds the projection outputs to BF16. Checkpoint storage stays BF16. Use the same kernel when comparing model variants since floating-point accumulation order can affect outputs. The run records the selected kernel.

For hybrid inference on the RTX 3050, install the CUDA build separately and use a PowerShell session for the hybrid runs:

```powershell
rtk proxy uv pip install --python .venv/Scripts/python.exe --target .cache/cuda-py --no-deps torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
$env:PYTHONPATH = (Resolve-Path .cache/cuda-py).Path
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --smoke --device hybrid --linear-kernel fp32-tiled
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --profile dev --batch-size 2 --device hybrid --linear-kernel fp32-tiled
```

Hybrid inference keeps BF16 weights resident across CPU and CUDA. The default GPU weight budget is 1,920 MiB. The evaluator checks for at least 768 MiB of free VRAM beyond that budget, and `--gpu-weights-mib` can lower it. Use the same budget for matched original and variant runs. Hybrid runs have their own protocol identity and record the per-layer placement and peak CUDA memory reserved by PyTorch.

`--validate-profiles` checks the local data and task wiring with a dummy model. Its output is a software check and produces no experiment record. Use `--batch-size` to test different memory and throughput settings between one and ten. The selected batch size is part of the comparison protocol.

Each attempt has its own directory. To continue an interrupted profile, pass `--resume-from RUN_ID` with the same profile, task list, and model. A new run uses the saved generation cache from the earlier attempt. For a Transformers-compatible variant with a complete asset manifest, use:

```powershell
rtk proxy .venv/Scripts/python.exe gptcompress/src/gptcompress/evaluation/run_evaluation.py --profile dev --model gptcompress/models/compressed/my-variant --parent-run ORIGINAL_RUN_ID
```

The variant manifest uses the same file list format as the original and adds a `compression` object. Provide the variant's identity, revision, and each file's relative path, size, and SHA-256. Non-Transformers representations require a backend before they can use this entry point.

`gptcompress/configs/benchmarks.json` records the additional benchmark sources, selected splits, and upstream scorer identifiers. The LongBench selection is NarrativeQA, Qasper, and TREC. The 2,048-token baseline context is too short for a comparable LongBench run. EvalPlus datasets are staged and candidate execution requires an isolated runtime. Docker Desktop has a CLI on this host and its Linux engine did not respond during verification. Evaluation for the additional suites is pending.

Once the Docker engine is available, build the pinned EvalPlus scoring environment and use a fresh run directory containing a `samples.jsonl` file in the EvalPlus sample format. Mount that directory to receive the scoring output. The example is for HumanEval+. For MBPP+ use `--dataset mbpp`, `MBPP_OVERRIDE_PATH=/bench/MbppPlus.jsonl`, and the corresponding data folder.

```powershell
rtk proxy docker build -f gptcompress/configs/evalplus.Dockerfile -t gptcompress-evalplus:0.3.1 .
$candidateRun = (Resolve-Path gptcompress/experiments/CANDIDATE_RUN_ID).Path
$benchmarkData = (Resolve-Path gptcompress/datasets/benchmarks/humaneval_plus/evalplus).Path
rtk proxy docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --memory 2g --cpus 2 --pids-limit 64 --tmpfs /tmp:size=512m --mount "type=bind,source=$candidateRun,target=/run" --mount "type=bind,source=$benchmarkData,target=/bench,readonly" -e HUMANEVAL_OVERRIDE_PATH=/bench/HumanEvalPlus.jsonl gptcompress-evalplus:0.3.1 --dataset humaneval --samples /run/samples.jsonl --parallel 2
```
