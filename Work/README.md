# GptCompress evaluation

Run these commands in `Work` with PowerShell. The workspace keeps full checkpoints in `models/original`, variants in `models/compressed`, immutable benchmark snapshots in `datasets/benchmarks`, and records in `experiments`. Keep compression calibration inputs in `datasets/calibration` and source them independently of the benchmarks.

```powershell
rtk proxy uv pip install --no-cache --python .venv/Scripts/python.exe --index https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
rtk proxy uv pip install --no-cache --python .venv/Scripts/python.exe -r requirements.txt
rtk proxy .venv/Scripts/python.exe -m nltk.downloader -d .nltk-data punkt_tab
rtk proxy .venv/Scripts/python.exe src/assets.py model
rtk proxy .venv/Scripts/python.exe src/assets.py data
rtk proxy .venv/Scripts/python.exe src/assets.py evalplus
```

The asset commands use public endpoints and pin each dataset to the revision recorded on its first fetch. They resume partial downloads. Check each manifest for available files, hashes, licenses supplied by the dataset card, and access errors. The all-datasets command exits nonzero if a source is blocked and still attempts the other sources.

```powershell
rtk proxy .venv/Scripts/python.exe src/evaluate.py --smoke
rtk proxy .venv/Scripts/python.exe src/evaluate.py --profile pilot
rtk proxy .venv/Scripts/python.exe src/evaluate.py --profile dev
rtk proxy .venv/Scripts/python.exe src/evaluate.py --profile heldout
rtk proxy .venv/Scripts/python.exe src/evaluate.py --report
rtk proxy .venv/Scripts/python.exe src/evaluate.py --validate-profiles
```

The runnable profiles cover GSM8K and IFEval. Development uses 10 fixed rows per task. Held-out evaluation uses the next 100 rows from the same seeded deterministic permutation. The ordered row IDs and dataset hashes are recorded in the run manifest. GSM8K retains the harness's five-shot protocol using training rows and its published answer filters. The headline score uses flexible extraction. IFEval uses its published strict prompt-level scoring and records all four scores. The profiles use Qwen chat formatting with thinking disabled, greedy decoding, batch size one, and 2,048 total tokens. Output limits are 192 tokens for GSM8K and 256 for IFEval. These are custom subset results under bounded decoding. Treat the reported output-length flag as a diagnostic based on re-tokenized text.

The local pilot uses the same 10 development IDs per task and scores with the same harness. It uses a batch of ten and 512 total tokens. GSM8K uses a zero-shot prompt asking for the final number and allows 24 output tokens. IFEval preserves the input instructions and allows 48 output tokens. The pilot measures the end-to-end workflow and carries its own comparison protocol. The disjoint held-out profile uses the larger budgets above.

The original model runs on CPU with its BF16 checkpoint and safetensors loading. Windows reports peak process working set. The report shows VRAM as unavailable for the CPU run. Elapsed times include generation and harness overhead, and the run records model load time in its overall wall time.

`--linear-kernel fp32-tiled` speeds up BF16 projections on the tested AVX2 CPU. The forward pass casts temporary weight tiles of up to 64 MiB and inputs to FP32 for the matrix product, then rounds the projection outputs to BF16. Checkpoint storage stays BF16. Use the same kernel when comparing model variants since floating-point accumulation order can affect outputs. The run records the selected kernel.

`--validate-profiles` checks the local data and task wiring with a dummy model. Its output is a software check and produces no experiment record. Use `--batch-size` to test different memory and throughput settings between one and ten. The selected batch size is part of the comparison protocol.

Each attempt has its own directory. To continue an interrupted profile, pass `--resume-from RUN_ID` with the same profile, task list, and model. A new run uses the saved generation cache from the earlier attempt. For a Transformers-compatible variant with a complete asset manifest, use:

```powershell
rtk proxy .venv/Scripts/python.exe src/evaluate.py --profile dev --model models/compressed/my-variant --parent-run ORIGINAL_RUN_ID
```

The variant manifest uses the same file list format as the original and adds a `compression` object. Provide the variant's identity, revision, and each file's relative path, size, and SHA-256. Non-Transformers representations require a backend before they can use this entry point.

`configs/benchmarks.json` records the additional benchmark sources, selected splits, and upstream scorer identifiers. The LongBench selection is NarrativeQA, Qasper, and TREC. The 2,048-token baseline context is too short for a comparable LongBench run. EvalPlus datasets are staged and candidate execution requires an isolated runtime. Docker Desktop has a CLI on this host and its Linux engine did not respond during verification. Evaluation for the additional suites is pending.

Once the Docker engine is available, build the pinned EvalPlus scoring environment and use a fresh run directory containing a `samples.jsonl` file in the EvalPlus sample format. Mount that directory to receive the scoring output. The example is for HumanEval+. For MBPP+ use `--dataset mbpp`, `MBPP_OVERRIDE_PATH=/bench/MbppPlus.jsonl`, and the corresponding data folder.

```powershell
rtk proxy docker build -f configs/evalplus.Dockerfile -t gptcompress-evalplus:0.3.1 .
$candidateRun = (Resolve-Path experiments/CANDIDATE_RUN_ID).Path
$benchmarkData = (Resolve-Path datasets/benchmarks/humaneval_plus/evalplus).Path
rtk proxy docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --memory 2g --cpus 2 --pids-limit 64 --tmpfs /tmp:size=512m --mount "type=bind,source=$candidateRun,target=/run" --mount "type=bind,source=$benchmarkData,target=/bench,readonly" -e HUMANEVAL_OVERRIDE_PATH=/bench/HumanEvalPlus.jsonl gptcompress-evalplus:0.3.1 --dataset humaneval --samples /run/samples.jsonl --parallel 2
```
