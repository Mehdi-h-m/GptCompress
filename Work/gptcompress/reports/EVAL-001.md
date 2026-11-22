# EVAL-001 worker handoff

The evaluation workspace is under `Work`. See [the runbook](../README.md) for current setup, asset downloads, evaluation profiles, variant evaluation, isolated coding evaluation, and report regeneration. Every command below that uses `src/evaluate.py` is an exact historical run command from before the source reorganization. Those old paths are provenance records, not current commands, and do not reproduce the past runs from the current source tree. Use `src/gptcompress/evaluation/run_evaluation.py` and `src/gptcompress/evaluation/prepare_assets.py` for current commands in the runbook.

## Assets

- Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c` is present. Its three original BF16 shards occupy **8,044,982,000 bytes** and match the upstream SHA-256 hashes. The 12 files listed in the model manifest occupy **8,060,925,056 bytes**. [Local manifest](../models/original/Qwen3-4B/manifest.json)
- Nine benchmark repositories downloaded successfully. The 40 downloaded files match their recorded SHA-256 hashes. The LongBench archive contains NarrativeQA, Qasper, and TREC. The tagged EvalPlus releases contain 164 HumanEval+ tasks and 378 MBPP+ tasks. [Benchmark configuration](../configs/benchmarks.json)
- GPQA Diamond returned HTTP 401 at the official pinned download URL. The other data downloads continued. [GPQA manifest](../datasets/benchmarks/gpqa_diamond/manifest.json)

## Verification

- `rtk proxy .venv/Scripts/python.exe -u src/evaluate.py --validate-profiles` passed with dummy responses. Each task has 10 development and 100 held-out IDs with no overlap. No compression calibration data was selected.
- Five existing report tests passed with Python's temporary directory set inside the workspace. They cover history preservation, deltas for comparable runs, missing scores, and escaping individual answers.
- `rtk proxy .venv/Scripts/python.exe -u src/evaluate.py --smoke` generated “Blue.” from the original BF16 model on CPU. Model loading took 1.95 seconds. Generation took 36.35 seconds for three tokens including EOS. Peak Windows process working set was 8,071,720,960 bytes (7.52 GiB). [Smoke metrics](../experiments/20261120T174224Z-e3633538/metrics.json)
- On the AVX2 host, a one-thread synthetic `700×1536` by `1536×1536` linear operation took 1.78 seconds through native BF16 and 0.058 seconds using FP32 casts with BF16 output. A separate synthetic check matched native BF16 outputs for the tested linear layer. These checks motivated the optional tiled CPU execution path run against the original model.
- `rtk proxy .venv/Scripts/python.exe -u src/evaluate.py --smoke --linear-kernel fp32-tiled` also generated “Blue.” in three tokens including EOS. Generation took 37.42 seconds with a peak working set of 8,193,634,304 bytes (7.63 GiB). This smoke check showed no end-to-end speedup. [Tiled CPU smoke metrics](../experiments/20261120T175641Z-519edcbb/metrics.json)

## Completed pilot

`rtk proxy .venv/Scripts/python.exe -u src/evaluate.py --profile pilot --linear-kernel fp32-tiled` completed in **9 minutes 22 seconds**, with a peak CPU process working set of **8,714,723,328 bytes (8.12 GiB)**. The pilot ran the original BF16 checkpoint on CPU with a batch of ten, a 512-token context, greedy decoding, and thinking disabled. [Run manifest](../experiments/20261120T175804Z-b6ee9b31/manifest.json)

| Benchmark | Score | Questions | Standard error | Task time |
| --- | ---: | ---: | ---: | ---: |
| GSM8K flexible exact match | 20% | 10 | 13.3 pp | 4 min 23 sec |
| IFEval strict prompt accuracy | 50% | 10 | 16.7 pp | 4 min 58 sec |

Eight GSM8K answers and nine IFEval answers were flagged near their respective 24- and 48-token output caps. No input was flagged as truncated. The short budgets and small samples limit what these scores say about general benchmark performance. [Metrics](../experiments/20261120T175804Z-b6ee9b31/metrics.json) · [Predictions](../experiments/20261120T175804Z-b6ee9b31/predictions.jsonl) · [Execution log](../experiments/20261120T175804Z-b6ee9b31/execution.log)

## Remaining evaluation

The [local report](index.html) shows the status and artifacts for each real run. The first development attempt and the first batched pilot were stopped while generating the first GSM8K batch. Their records are preserved. Docker Desktop's Linux engine was unavailable during the check, so the staged EvalPlus inputs and Docker configuration have no coding score. Larger profiles and additional suites have no measured results.
