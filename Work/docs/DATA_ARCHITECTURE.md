## DATA-RESEARCH-001 - Factorization and distillation data proposal

Status: Research proposal, 2026-11-22; no implementation assignment. The user requests a researched collection/scraping pipeline and quantity estimates, prioritizing preservation of as many original Qwen3 capabilities as possible. Quantities and mixture weights below are proposed experiments, not measured recovery thresholds.

This standalone architecture document was requested on 2026-11-22. The [compute and storage orchestrator design](drm/docs/ORCHESTRATOR_ARCHITECTURE.md) describes provider integrations, automatic placement and the experiment budget. Infrastructure suggestions below remain provisional.

Objective and scope

- Start with the existing pinned original Qwen3-4B teacher and a factorized student, preserving tokenizer/template identity. Input-aware factorization targets `||WX - ABX||_F^2`; distillation is a separate recovery stage requiring teacher supervision. Stronger external teachers change the question toward capability transfer.
- Cover knowledge, instruction following, multi-turn conversation, code, mathematics, languages, both thinking modes and context lengths. Complete retention cannot be guaranteed; compression rank, coverage, compute and acceptable loss remain coupled decisions. Average benchmark scores do not establish retention of every capability.
- This request authorizes research and the proposal, not implementation, collection/training runs, paid compute or new model downloads. Existing assignments retain their specific constraints.

Proposed data pipeline

1. Pin a source registry: revisions, subsets, URLs, license/terms references, retrieval times, token quotas and checksums. Preserve per-source licensing within mixed datasets. Prefer public dataset streams and repository/documentation exports; use an allowlisted crawler for coverage gaps.
2. Stream selected training shards with seeded shard selection and buffered shuffle. Do not take only the first records; buffered shuffle is not globally uniform. Stop at accepted Qwen-token quotas and explicit scan/download/disk limits. Make stages resumable.
3. Discover crawl URLs through sitemaps; respect robots/source terms, rate-limit per host, retry with backoff, and use Trafilatura for extraction. Preserve code indentation, formulas and headings. Record failures; avoid private/login-only sources and do not execute downloaded code during ingestion.
4. Normalize conservatively, identify languages, remove boilerplate/repetitive spam/malformed content and detected exposed secrets/personal identifiers. Use task-specific filters so code/math survive prose quality filters. Inspect samples in each bucket.
5. Deduplicate exact hashes and near-duplicate clusters across sources. Exclude overlaps with benchmark questions/solutions and known derivatives where detectable. This is best-effort decontamination, not proof against every paraphrase. Split by source document/conversation/repository/problem family and duplicate cluster BEFORE chunking or generating derivatives; preserve upstream held-out splits.
6. Initial pool: approximately 100M accepted Qwen tokens; roughly 90% training material, 5% validation, 5% locked corpus holdout, with grouped splitting taking precedence over exact ratios. Allocate calibration separately within the training partition, disjoint from recovery training for the initial comparison. Existing external benchmark splits remain separate.
7. Retain raw text/messages, uint32 token IDs, masks, parent/source/split IDs, task/language/length/mode labels, revisions and checksums in bounded shards. Preserve conversation boundaries and assistant loss masks; do not treat truncated reasoning as completed answers. Use correct boundaries/masks when packing unrelated documents.
8. Export calibration, recovery, validation and holdout manifests plus accepted/rejected token counts, coverage, duplication, contamination and truncation reports. Clean data: Parquet; metadata/index: SQLite or equivalent. DataTrove supplies reusable filtering/deduplication components; verify Windows compatibility before committing to its executor.

Candidate token mixture

- 40% general/educational prose: FineWeb-Edu English plus FineWeb-2 language subsets.
- 25% instructions/dialogue: selected SmolTalk components, including constraints, rewriting, summaries, multi-turn and structured output.
- 20% code: SmolTalk coding component plus explicitly licensed repository/documentation material, grouped by repository.
- 15% math/reasoning: filtered OpenR1-Math problems and appropriate SmolTalk components after benchmark/cross-source decontamination.
- These are starting token shares. Language/length quotas are separate axes. Include English, Chinese, Arabic, French and additional sampled languages; a small corpus cannot substantiate retention across every supported language. Preserve both thinking modes as labeled strata. Existing dataset answers are external-teacher outputs, not the pinned Qwen's outputs.
- Start calibration with 2048-token contexts; add 4096/8192 and longer strata when feasible. Short-context experiments cannot establish long-context retention.

Quantity and teacher-supervision proposal

- Smoke: 32 x 512 calibration tokens and 0.1-0.5M recovery tokens.
- Pilot: 512 x 2048 calibration tokens = 1,048,576; 5-10M unique recovery tokens. Test nested calibration sizes of 128, 512 and 2048 windows (0.26M, 1.05M, 4.19M tokens), stratified by workload.
- Follow-up: 50-100M recovery tokens if the pilot supports continuing. Expand the source pool beyond 100M as necessary to preserve independent calibration/holdouts. Consider 0.5-1B only if learning curves and compute justify it. Published broad distillation studies can use tens/hundreds of billions; these budgets do not guarantee broad recovery.
- Count unique tokens separately from epoch token presentations; report input, assistant, thinking, supervised and padding tokens. Token counts take precedence over example counts.
- Teacher forcing on clean sequences provides logits without generating every token autoregressively. Add 2000-5000 Qwen-generated conversations for the pilot, both modes, scaling toward 20k-50k only after throughput/quality measurements. These contribute to the recovery token budget. Record generation settings, truncation and provenance; verify math/code answers where possible.
- Consider block-local output/feature matching to train factors within local memory limits. Global logit distillation is a separate, more demanding step. Local matching does not guarantee whole-model recovery; assess accumulated compressed-prefix drift.
- For offline global KD, test top-64/top-128 probabilities with token indices, temperature/normalization metadata and residual probability mass. This is an approximate/coarsened objective, not exact full-vocabulary KL. Compare to full distributions on a bounded diagnostic sample. Cache keys include teacher identity, exact sequence/masks, template and arithmetic.
- Stream second moments and bounded block targets; avoid storing every activation. Document covariance sketches, damping and approximations, if used.

Resource and decision criteria

- Existing 4GB VRAM / approximately 16GB RAM host can collect/tokenize data. Full teacher-plus-student training is not a realistic default. Benchmark sequential teacher/student passes, CPU staging and block-local training; do not promise global KD fits. No paid compute is authorized.
- Arithmetic estimates excluding overhead: 100M uint32 IDs = 400MB. Qwen vocabulary 151936 means 10M positions of full FP16 logits = 3.04TB. Top-64 uint32 IDs + FP16 values = 3.84GB plus residuals/metadata. One 1M-token x 2560-hidden FP16 activation buffer = 5.12GB.
- Proposed initial data/cache cap: 20GB excluding checkpoints. Bound raw retention; measure bytes/token on a 1M-token ingest and re-estimate before scaling. Benchmark teacher forcing, generation and training throughput independently before time estimates.
- Compare original, weight-only factors, input-aware factors and input-aware factors plus distillation at matched parameter budgets/protocols. Use development data to select ranks/budgets, then one frozen candidate on locked holdouts. Report per-capability deltas, uncertainty and multiple calibration seeds where feasible.
- Separate the effects of more unique data from more optimizer steps. If recovery stalls, investigate less aggressive/heterogeneous ranks instead of assuming more scraping resolves a capacity deficit. No final compression ratio or acceptable quality loss has been selected.

Data contracts and reproducibility

| Artifact | Required metadata |
|---|---|
| Source | Stable ID, URL, immutable revision/snapshot, license/terms, selected subsets, quotas |
| Document | Source/upstream IDs, parent ID, retrieval time, raw/clean hashes, text/messages, language, primary task, quality flags |
| Partition | Origin group, duplicate cluster, split, seed, algorithm version |
| Token sequence | Parent IDs, token IDs, tokenizer revision, template hash, attention/position information, loss masks, offsets, mode, truncation status |
| Teacher response | Parent prompt, teacher revision, generation settings/seed, mode, finish reason, verification result |
| Activation target | Teacher identity, sequence selection, module/capture point, positions/masks, tensor/statistic type, dtype, normalization and approximations |
| Logit target | Sequence ID, teacher identity, supervised positions, next-token alignment, temperature, indices/values, normalization, residual mass |
| Release | Schema version, source revisions, code/config hashes, seed, shard paths/checksums, counts, coverage and validation report |

Text releases and model-specific targets are separate. Changing the teacher, tokenizer/template, sequence, masks, capture location or arithmetic invalidates affected targets. Teacher logits at position t supervise the next token; verify this alignment explicitly. Top-k values must state whether normalization is full-vocabulary or retained-only.

All translations, generated questions and answers inherit their parent's split. Recheck generated content for duplicates and benchmark overlap. Assign one primary quota bucket per sequence to prevent double-counting; track secondary labels separately. Cap dominant domains and repetitive templates. Overlapping windows from one document do not constitute independent coverage.

Proposed artifact layout (relative to Work)

```text
configs/data/                 Source registry and pipeline settings
datasets/raw/                 Bounded snapshots and acquisition manifests
datasets/clean/               Canonical text/messages and provenance
datasets/index/               Duplicate groups and split assignments
datasets/releases/<id>/       Immutable tokenized shards and release manifests
datasets/calibration/<id>/    Calibration selections
datasets/teacher/<id>/        Model-specific supervision artifacts
reports/data/<id>/            Coverage, quality and resource evidence
experiments/<id>/             Compression and distillation results
```

These directories are proposed, not created by this document. Reference shared immutable shards rather than copying payloads. Validate schema, counts and checksum before committing each shard and advancing a resumable stage checkpoint. Incomplete artifacts must not appear ready for training. Preserve artifacts referenced by experiments when designing cache eviction.

Acceptance gates

| Stage | Required evidence |
|---|---|
| Acquisition | Source identity/provenance, bounded use, working resume |
| Extraction | Inspected examples across sources/tasks/languages, preserved code/math formatting |
| Deduplication/splits | Duplicate counts/thresholds, no known origin-group leakage, benchmark exclusion report |
| Tokenization | Correct template/boundaries/masks, measured lengths, explicit truncations |
| Teacher targets | Correct alignment/cache identity, finite values, probability/tail consistency |
| Release | Matching checksums/counts, achieved versus requested coverage, documented exclusions |
| Experiment | Matched baselines, per-capability deltas/uncertainty, resource measurements, locked holdouts |

Validate a small release, then roughly 1M accepted tokens, before scaling collection. Report scanned/accepted tokens, rejection reasons, duplication, source diversity and bytes at every stage. Automated checks cannot prove absence of every paraphrase or quality defect.

Open decisions

- Compression budget and acceptable per-capability loss.
- Exact language, length and thinking-mode quotas.
- Storage backend, target compression, cache retention and eviction.
- Compute availability, execution strategy and measured throughput.
- Block-local/global distillation choice, loss weights and precision.
- Final source revisions/allowlist and filtering thresholds.

The earlier 20GB cap and top-k caches are candidates, not validated infrastructure choices. Collection, activation capture, teacher target production and training should remain independently resumable. Block-local targets must record whether inputs originate from the original or compressed prefix; local reconstruction and accumulated divergence are distinct measurements.

Primary references checked

- SVD-LLM calibration example (256 x 2048; LLaMA, not a Qwen guarantee): https://github.com/AIoT-MLSys-Lab/SVD-LLM/blob/main/compress_llama.sh
- Qwen modes/configuration: https://huggingface.co/Qwen/Qwen3-4B and https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/raw/main/README.md
- SmolTalk (component-specific licenses): https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- OpenR1-Math: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- Streaming: https://huggingface.co/docs/datasets/stream
- Crawling/extraction: https://trafilatura.readthedocs.io/en/latest/crawls.html
- Deduplication/processing: https://github.com/huggingface/datatrove
- Broad distillation scale (different models/method): https://arxiv.org/html/2408.11796v1


