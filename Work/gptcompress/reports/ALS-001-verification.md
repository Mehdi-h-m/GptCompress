# ALS-001 independent verification

The master’s delegated worker owns the conversion, evaluation, and final ALS-001 handoff. Its jobs were already active when this worker found the overlap. The shared checkout briefly contained competing edits to `src/compress.py` and `src/evaluate.py`. This worker restored the delegated implementation from its edit history, removed its own overlapping code, and verified that both entry points compiled. The started jobs were left in place. The commands below are preserved as historical verification records; current entry points are `src/gptcompress/compression/compress_model.py` and `src/gptcompress/evaluation/run_evaluation.py`.

All nine tests passed on the restored checkout, including synthetic checkpoint reload, tied weights, deficient and zero inputs, and report history preservation.

```powershell
rtk proxy .venv/Scripts/python.exe -m py_compile src/compress.py src/evaluate.py src/gptcompress/factorization.py
rtk proxy .venv/Scripts/python.exe -c "import pathlib,tempfile,unittest,torch; tempfile.tempdir=str(pathlib.Path.cwd()); torch.set_num_threads(2); suite=unittest.defaultTestLoader.discover('tests'); raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())"
```

## Real weight probe

`experiments/ALS-001-probe-r256/progress.json` records all six full-size matrices and 20 ALS iterations per matrix. The probe took 81.88 seconds and peaked at 1,357,852,672 bytes of Windows process working set. Each result below is a fraction of the original matrix’s squared Frobenius norm. The SVD reference uses the same full weight at rank 256.

| Block | Projection | ALS after BF16 storage | Exact rank-256 SVD | Gap in percentage points |
| --- | --- | ---: | ---: | ---: |
| 0 | MLP down | 0.756618 | 0.755951 | 0.0667 |
| 0 | Attention key | 0.433112 | 0.432443 | 0.0669 |
| 18 | MLP down | 0.731610 | 0.730940 | 0.0670 |
| 18 | Attention key | 0.271154 | 0.270826 | 0.0328 |
| 35 | MLP down | 0.776830 | 0.775973 | 0.0857 |
| 35 | Attention key | 0.405252 | 0.404664 | 0.0587 |

Independent checks of the in-progress MLP conversion found both BF16 factors with the expected rank and input/output dimensions in the first projection shards. Six untouched normalization tensors from blocks 0 and 1 matched the original bytes. The saved records for the first 11 converted matrices each reported fewer factor parameters than original parameters. A complete checkpoint and model inference remain to be verified.

## Planned storage

The original checkpoint directory contained 8,060,927,654 bytes across its files, including the manifest. These estimates use the original safetensors shapes and two bytes per BF16 weight. Final serialized size also includes metadata, shard headers, and tokenizer assets.

| Rank | Target | Eligible matrices | Factor parameters | Estimated tensor payload bytes |
| ---: | --- | ---: | ---: | ---: |
| 256 | MLP | 108 | 339,738,624 | 3,345,218,560 |
| 256 | Attention | 144 | 188,743,680 | 6,534,986,752 |
| 256 | Combined | 252 | 528,482,304 | 1,835,269,120 |
| 128 | Combined | 252 | 264,241,152 | 1,306,786,816 |
| 512 | Combined | 252 | 1,056,964,608 | 2,892,233,728 |

At the checkpoint sampled for this report, rank-256 MLP conversion had completed 11 of 108 matrices in 268 seconds. The median observed solver time was 18.24 seconds, and peak working set was 1,357,795,328 bytes. The first conversion had an estimated 30–40 minutes remaining at that point on this machine.

## Evaluation status

Original development attempt `20261120T231535Z-d2b30f4e` used batch 10, four CPU threads, the FP32-tiled linear kernel, BF16 storage, and the existing development budgets. The delegated worker stopped the attempt at 97.8% host RAM use while conversion was also active. Neither task had produced a score. A sequential original/variant comparison can use a smaller matched batch. Variant development scores, candidate selection, and the held-out comparison remain pending.
