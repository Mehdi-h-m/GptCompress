# GptCompress worker requirements

## DATA-RESEARCH-001 - Factorization and distillation data proposal

App design: [ORCHESTRATOR_ARCHITECTURE.md](drm/docs/ORCHESTRATOR_ARCHITECTURE.md) specifies provider onboarding, automatic compute and data placement, scientific partitioning and result merging, recovery, and the experiment spending ceiling. This is a design deliverable.

Full documentation: [DATA_ARCHITECTURE.md](docs/DATA_ARCHITECTURE.md), created on 2026-11-22. Storage and compute optimization remain the next design step.

Cloud research constraint (2026-11-22): the user wants most data storage and compute off the local machine, using free tiers/new-user credits where feasible, with a total out-of-pocket budget of at most USD 7-10 for this experiment (not a monthly subscription budget). Research provider eligibility, real GPU availability, persistence, transfers and billing exposure before selecting a deployment. This request authorizes research only; account creation, upgrades, uploads and paid resource launches have not been requested. Candidate design: Hugging Face private storage plus Kaggle notebooks; optional Colab, Modal credits and Google Cloud trial subject to verified account access. Advertised quotas are not measured account entitlements. Update the architecture with an accepted deployment design after provider selection.

Workspace layout (reorganized 2026-11-22): `Work/gptcompress/` holds the compression method — src, tests, configs, reports, experiments, datasets, models. `Work/inspector/` holds the VIS-001 desktop app — tests, reports, inspection recordings, and dist builds; its code remains the `gptcompress.inspector` package under `Work/gptcompress/src/gptcompress/inspector/`. `Work/drm/` holds the Distributed Resource Manager app. `Work/docs/` holds cross-project architecture documents. `Work/reports/` retains only the generated dashboard. Historical assignment records below keep the paths as they were when the work was done.

Status: Research proposal, 2026-11-22; no implementation assignment. The user requests a researched collection/scraping pipeline and quantity estimates, prioritizing preservation of as many original Qwen3 capabilities as possible. Quantities and mixture weights below are proposed experiments, not measured recovery thresholds.

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

## VIS-001 — Local model inspection and version comparison app

Status: Assigned on 2026-11-21. The user authorizes implementation and local verification of the complete app described below.

Objective: Build a local interactive app to explore LLM architecture, inspect weights, record inference activations, and compare any two selected model versions. Initial verified support must cover the existing original Qwen3-4B and ALS rank-256 MLP checkpoint. Design extensible adapters for additional architectures and representations. Claim support only for verified cases.

Ownership and dependencies: The visualization worker owns necessary app/backend implementation, adapters, focused tests, app configuration, documentation, and new recording/report artifacts under Work/. Reuse the existing checkpoint loaders, factorized modules, manifests, experiment records, and evaluation logic. Read root AGENTS.md, Work/AGENTS.md, this brief, and RTK.md. Check for CodeGraph before code navigation. Coordinate with active workers before overlapping edits. Preserve existing results, unrelated user edits, and original checkpoints. The master owns this brief. Do not implement new compression methods as part of this assignment.

Architecture requirements

1. Provide a local browser interface and local Python inspection/inference service, with persistent model-version and recording metadata. Choose a maintainable frontend stack after inspecting existing dependencies. Use background jobs with progress, cancellation, recoverable errors, and bounded resources for expensive inspection or inference. The app must remain usable while a job runs. Provide a documented launch command and verify the normal user workflow.
2. Define an adapter contract for model metadata, hierarchical modules, tensor inventory, checkpoint loading, semantic component mappings, and supported capture locations. Provide a generic tensor/module inspection fallback and concrete original-Qwen3 and existing factorized-Qwen3 support. Distinguish a module hierarchy from verified execution/dataflow edges. Missing adapter capabilities must be explicit in the UI.
3. Maintain a model-version registry. Each entry has a stable ID, display name, checkpoint path and immutable identity, model family, parent version when known, transformation settings, representation details, tokenizer identity, and links to available experiments. Discover/import existing manifests. Support adding local versions through the app. Preserve provenance and avoid duplicating checkpoint payloads.
4. Version A and Version B are independently selectable. Neither is hardcoded to the original model. Support reversing the pair and comparing any registered supported pair. Record component correspondences through transformations. For current factors, map A and B back to the logical original projection. Define mapping metadata for future channel/head/block pruning, including retained indices and removed components. Do not fabricate mappings for existing artifacts whose provenance lacks them. Distinguish unmatched, incompatible, removed, and unavailable components.

User-facing capabilities

5. Architecture explorer with zoom/pan, expandable transformer blocks, attention/MLP/normalization/projection detail, breadcrumbs, and selection. Show shapes, parameter counts, dtype, and actual storage where available. Count shared tensors once for model totals. Display factorized projection structure and its logical relationship to W. Use clear layouts and purposeful labels.
6. Matrix inspector with bounded tiled/aggregated heatmaps, detailed slices on demand, histograms, extrema, mean/std, norms, row/column summaries, and optional singular spectra. Label sampling, aggregation, approximation, precision, and computation costs. Compute expensive statistics on demand and cache against checkpoint identity and algorithm settings. Include reconstruction error and retained spectral energy where a corresponding source matrix exists. Avoid full dense reconstruction of all factors or full matrix transfers to the browser.
7. Inference recorder with prompt entry, version selection, token budget, chosen modules/token positions, and capture settings. Capture summaries by default and selected raw slices only on request. Persist token IDs/text, prefill/decode phase, positions, model/checkpoint identity, capture locations, settings, execution arithmetic, and recording status. Capture hook inputs/outputs without altering model behavior. Remove hooks and release resources after success, error, or cancellation. Verify capture-enabled inference against capture-disabled inference under deterministic settings.
8. Compare model-level storage, existing benchmark results, and available runtime/memory measurements. Show benchmark deltas only for matching protocols and question IDs. Label missing measurements and preserve sample counts. Runtime instrumentation overhead must be identified. Provide per-block activation comparison, matrix comparisons, and aligned next-token probability differences where valid. Use identical fixed token sequences for paired diagnostic forward passes. Check tokenizer/vocabulary compatibility before token/probability alignment. Do not interpret divergent free-generation positions as equivalent contexts.
9. Distinguish local approximation error from accumulated end-to-end divergence. For a mapped original/factorized projection, feed the same captured input into both compatible modules and measure relative output error, with a defined zero-denominator policy. Separately compare matched block outputs from whole-model forward passes. For pruning or shape changes, use explicit channel mappings for elementwise comparison or compare compatible block boundaries. Never silently crop, pad, or subtract incompatible tensors. Show the basis of each comparison in the UI.
10. Offer a layer overview that highlights measured divergence and lets users drill into a selected layer, matrix, or token. Display cosine similarity and relative error where defined, activation norms/outliers, and selected output distributions. State that observed differences are diagnostic evidence, not causal proof. Save and reopen recordings and comparison selections without rerunning models.

Resource and scope requirements

- Existing Windows host, RTX 3050 4 GB VRAM and approximately 16 GB RAM. Run original and compressed models sequentially. Use CPU staging, bounded activation buffers, limited concurrent jobs, chunked statistics, and explicit capture budgets. Inspect available disk space before recordings. Do not retain both complete models in memory.
- Bind the service to localhost. Keep prompts, tensors, and checkpoints local. Reuse installed libraries where practical and install needed dependencies locally. No paid compute, external publication, Git push, or large new model downloads are authorized here.
- Keep model inspection read-only. Handle unknown model types gracefully. Future pruning compatibility belongs in adapter/mapping design; current deliverables must operate on the actual available models. Unsupported attention-weight capture under optimized kernels must be disclosed rather than silently changing inference semantics.

Delivery and acceptance

- Complete usable app, launch instructions, adapter/version/recording format documentation, and Work/inspector/reports/VIS-001.md with changed files, exact verification commands, artifacts, measured memory/runtime, and known limitations.
- Verify architecture navigation and matrix inspection for original and factorized Qwen3; confirm displayed sizes/shapes against real artifacts. Demonstrate persisted activation recording and A/B selection, swap, reopen, and a real aligned comparison on a short prompt. Produce a diagnostic walkthrough identifying measured divergence locations without claiming causality.
- Focused tests must cover stable version identity, factor mappings, incompatible shapes/tokenizers, missing data, finite/zero-value statistics, save/reopen, and hook cleanup/capture parity. Test pruning correspondence with clearly labelled small synthetic fixtures if no actual pruned model exists.
- Verify the UI in a browser at desktop dimensions, including background progress/errors, actual rendered plots, and working interactions. Use a modest real-model token budget for acceptance. Synthetic data must never appear as measured Qwen results.
- Send an early inventory/feasibility assessment, milestone updates, and a final evidence handoff. Recover within scope autonomously and report concrete blockers. A design mockup alone does not complete this assignment.

## ALS-001 — Fixed-rank weight-only compression

Status: Assigned on 2026-11-21 at the user's request. This assignment authorizes implementation and local experiments.

Objective: Factorize Qwen3-4B attention and MLP projection weights using alternating least squares (ALS), minimizing squared Frobenius reconstruction error with a user-selected fixed rank. Use no activation calibration or gradient training. Establish measured storage/quality trade-offs.

Ownership: The assigned worker owns necessary implementation, configuration, focused regression tests, and runbook changes under Work/, plus new compressed artifacts, experiments, and reports. Preserve existing runs and unrelated changes. The master owns this requirements file. Coordinate with any active worker before overlapping edits.

Dependencies: Reuse the pinned original checkpoint, existing evaluation entry points, manifests, and development/held-out splits from EVAL-001. Read Work/AGENTS.md and RTK.md first. Inspect current state and use CodeGraph first if indexed.

Implementation requirements

1. Expose rank, seed, convergence tolerance, iteration limit, and target projections as explicit configuration. Begin with rank 256, then compare 128 and 512. These are proposed experimental settings, not quality guarantees. Factorize only matrices satisfying r(m+n)<mn and record skipped matrices. Cover MLP gate/up/down and attention q/k/v/o projections. Preserve biases, nonlinearities, residual connections, attention configuration, embeddings, output head, and tied parameters.
2. Initialize reproducibly and alternate stable least-squares solves for A and B. Avoid explicit matrix inversion and poorly conditioned normal-equation solves. Record relative squared Frobenius error, convergence history, iteration count, elapsed time, and resource measurements. Handle rank-deficient factors, nonfinite values, and failed solves explicitly. Evaluate residuals in bounded chunks and use sufficient numerical precision. Record any regularization because it changes the objective. Avoid an initialization that makes the ALS experiment merely an already-optimal SVD solution.
3. Validate first on synthetic matrices with known rank and small representative real matrices. Sample eligible projections in early, middle, and late blocks. Compare to truncated SVD at matched rank where feasible. Distinguish exact SVD references from approximate references. Explain convergence gaps without claiming ALS can outperform the fixed-rank Frobenius optimum.
4. Stream checkpoint matrices and process sequentially, keeping bounded workspaces. Save factors in a documented precision and measure reconstruction again after storage conversion. Implement direct A(Bx) execution, portable save/reload, complete model manifests, and compatibility with the normal variant evaluation entry point. A standalone checkpoint must include all required unmodified weights and assets. Verify logits or outputs before and after reload and original behavior on untouched modules.
5. Progress through rank-256 MLP-only, attention-only, and combined variants, then combined variants at ranks 128 and 512. Reuse factors for identical matrices/settings where possible. Record failed, skipped, or resource-limited experiments. Estimate disk and peak memory before each stage. If representative checks reveal a correctness problem, resolve it before full conversion. Report substantial runtime/resource obstacles promptly with a bounded recovery proposal.
6. Establish an original-model development baseline with the existing longer development budgets, then evaluate variants with identical question IDs, kernel, batch size, prompts, precision policy, and decoding settings. Preserve the pilot separately. Record output-cap diagnostics. Select a candidate using development results, explicitly report the selection rule, and evaluate that candidate and original on the existing disjoint held-out profile. Do not tune settings using held-out results or assert an acceptable quality-loss threshold on the user's behalf.
7. Extend existing reports to compare actual serialized bytes including metadata, factor parameter counts, per-task score deltas, timing, RAM/VRAM, and matrix reconstruction error. Keep hardware placement and inference arithmetic explicit. Smaller size alone does not establish a speedup. Produce Work/gptcompress/reports/ALS-001.md with exact commands, changed files, artifact paths, measured results, failures, and limitations.

Acceptance checks: Stable solver behavior on known-rank and deficient inputs; factor storage truly smaller for eligible matrices; all projections preserve shapes; save/reload parity; real compressed-model inference through the existing evaluator; matched original/variant development results; final held-out comparison when feasible; report history preserved and reproducible configuration recorded. Any uncompleted criterion needs an explicit cause and status.

Resource limits: Existing Windows host, RTX 3050 4 GB VRAM, approximately 16 GB RAM. CPU staging and sequential processing are required when memory demands exceed GPU capacity. Account for solver workspaces and avoid full duplicate models. Existing project-local dependencies may be extended as needed. No paid services, external publication, Git push, or larger-model downloads are included.

Worker handoff: Send an early feasibility assessment, meaningful milestone updates, and a final evidence report. Continue recoverable work autonomously. Implementation alone does not complete the experiment assignment.

Status: Research brief and architecture constraints. Individual implementation work requires a concrete assignment from the master agent.

## 1. Project objective

Investigate how far an existing open-weight language model can be compressed while preserving useful behavior. The deliverable is a smaller executable model representation and evidence about the size/quality tradeoff.

Maximum compression is not yet a complete optimization objective: acceptable quality loss, target workload, and deployment context length remain open. Workers must not invent final thresholds or claim a method is successful solely because it reduces parameter count.

The user wants theoretical approaches and their combinations investigated, rather than a project consisting only of running an existing quantizer.

## 2. Established context

- Primary model: `Qwen/Qwen3-4B`.
- Stretch model: `Qwen/Qwen3-8B`, only when included in an assignment.
- Available hardware: NVIDIA RTX 3050 with 4 GB VRAM; approximately 16 GB system RAM observed on the current machine.
- Host environment: Windows and PowerShell.
- Working project: `C:/Users/djame/Downloads/GptCompress`.
- Preserve the existing `New folder` directory and any unrelated user files.
- Updated research direction (2026-11-22): the user explicitly wants to investigate input-aware matrix factorization plus distillation, prioritizing preservation of as many original Qwen3 capabilities as possible. Distillation is now in scope for proposals. The current request authorizes research and a data-pipeline proposal, not implementation, collection, training runs, paid compute, or new model downloads. Earlier experiment assignments retain their specific constraints.
- Do not quietly introduce gradient-trained rotations, rounding parameters, codebooks, adapters, or recovery stages. Identify such dependencies in a proposed method and report them to the master before making them part of the experiment.
- Calibration-free, post-training, and gradient-free describe different properties. Label methods accurately.

## 3. Roles

### Master agent

- Own research direction, architecture, written requirements, task decomposition, and acceptance decisions.
- Give workers bounded assignments with owned files, dependencies, deliverables, and checks.
- Resolve conflicts between workers and revise requirements when decisions change.
- Review changes and experiment evidence without writing implementation code or tests.
- Report findings and limitations to the user.

### Worker agents

- Read this file before starting an assignment.
- Implement only the assigned deliverable, following applicable repository instructions.
- Inspect the existing project before assuming its structure or creating overlapping components.
- Make routine implementation decisions autonomously within the assignment.
- Report missing prerequisites or incompatible assumptions promptly; continue independent work where possible.
- Return concrete evidence, including changed paths, commands actually run, results, and unresolved limitations.
- Do not claim a result from a proposed experiment, a paper's benchmark, or a synthetic test as a measured result on the user's model.

## 4. Candidate research directions

These are candidates, not an order to implement all of them.

### Activation-aware low-rank factorization

Approximate each selected weight matrix W with factors A and B. With column-oriented calibration inputs X, target output error `||WX - ABX||_F^2`, rather than only weight reconstruction error.

- Candidate targets: MLP up/gate/down projections and attention query/key/value/output projections.
- Allow ranks to vary by matrix or layer.
- Count both factors and any associated metadata in storage measurements.
- Preserve nonlinearities and gates; do not collapse an entire nonlinear MLP into one linear matrix.
- Factorization of projection weights alone does not imply smaller attention head dimensions or a smaller KV cache.
- Relevant references: [ASVD](https://arxiv.org/abs/2312.05821) and [SVD-LLM](https://arxiv.org/abs/2403.07378).

### Structured pruning with analytic compensation

Candidate units: MLP intermediate channels, attention heads with their dependencies, whole transformer blocks, or representation dimensions through a compatible transformation.

- Remove dependent rows and columns consistently.
- Respect grouped-query attention, residual paths, normalization, positional encoding, and model configuration constraints.
- Where justified, compensate for removed contributions using calibration statistics or analytic reconstruction.
- Zeroing weights in a dense tensor is not a physical size reduction.
- Relevant references: [FLAP](https://arxiv.org/abs/2312.11983), [SLEB](https://arxiv.org/abs/2402.09025), and [SliceGPT](https://arxiv.org/abs/2401.15024).

### Low-precision weights with compact corrections

Investigate `W ≈ Q + AB`, where Q is quantized and AB corrects important residual error. A sparse residual S may be considered in a separately assigned extension.

- Include the quantized matrix, factors, scales, codebooks, and sparse indices in the byte budget.
- Compare against ordinary quantization at matched storage budgets where possible.
- Separate analytic decomposition from optional gradient-based recovery in published implementations.
- Relevant reference: [CALDERA](https://arxiv.org/abs/2405.18886).

### Shared bases and tensor representations

Investigate shared bases across compatible projections/layers, or Tensor Train, Tucker, and Kronecker representations where justified.

- Shared shape is not evidence that two matrices have interchangeable behavior.
- Preserve layer-specific information where needed.
- Count each shared object once in serialized storage and identify any runtime duplication.
- A compact representation must be executed without permanently materializing the entire dense model to substantiate a runtime-memory claim.
- Relevant references: [Basis Sharing](https://arxiv.org/abs/2410.03765) and [TensorGPT](https://arxiv.org/abs/2307.00526).

## 5. Proposed system architecture

Separate the following responsibilities. The master will specify concrete modules and interfaces in implementation assignments.

1. Model access: pinned source model revision, tokenizer, configuration, and memory-bounded tensor/block loading.
2. Calibration: reproducible sample selection, preprocessing, activation statistics, and bounded storage of intermediate data.
3. Compression methods: independent transformations with explicit supported modules, memory requirements, configuration, and metadata.
4. Executable representations: dense, factorized, quantized, sparse, or shared modules as required by the assigned method.
5. Serialization: persist the representation and reconstruction/runtime metadata; reload without requiring the original checkpoint unless explicitly labeled as a delta artifact.
6. Evaluation: common data and generation settings, quality metrics, physical size, and resource measurements.
7. Experiment records: model revision, method settings, data splits, software versions, seeds, measurements, failures, and artifact paths.

Use the original high-precision model as the scientific reference. Already-quantized checkpoints may serve as comparison baselines; do not silently substitute them for the original when testing factorization or pruning.

## 6. Resource requirements

- The 4B high-precision model will not fit wholly in 4 GB VRAM. Do not design around loading the entire model onto the GPU.
- Prefer block/matrix processing, CPU staging, checkpoint streaming, and bounded calibration batches where the method permits.
- Account for SVD workspaces, covariance matrices, factors, activation buffers, model copies, CUDA overhead, and other applications using memory.
- Estimate peak memory before expensive decompositions. A matrix fitting in memory does not establish that its decomposition will fit.
- Separate method limitations from limitations of a particular reference implementation.
- A large checkpoint download or full benchmark is not a side effect of imports, setup, or ordinary tests. Perform model-scale work when the assignment includes it.
- Do not switch runtimes, provision paid compute, or expand to larger models as an unreported workaround.

## 7. Evidence and acceptance requirements

- Maintain distinct calibration, development/selection, and final test data. Do not use final test results to select compression settings.
- Report accuracy changes as absolute percentage points where appropriate. Report relative changes separately and label them.
- Use perplexity as one diagnostic, not as proof that instruction following or task capability is preserved.
- Record actual serialized bytes, including all required auxiliary data. Distinguish parameter count, nominal precision, effective bits per original weight, disk footprint, RAM, and VRAM.
- Specify context length, batch size, token limits, generation mode, device placement, and timing conditions when reporting performance.
- Compare variants under equivalent evaluation settings. Keep output token budgets and thinking/non-thinking behavior consistent.
- A smaller checkpoint does not establish a faster model. Report speed only when measured.
- Verify relevant shape/configuration dependencies and save/reload behavior for changed representations.
- Use focused mathematical and integration checks appropriate to the change. Synthetic checks may establish correctness of an operation; they do not establish compression quality on Qwen.
- Record failed or out-of-memory experiments rather than omitting them from the conclusions.

## 8. Worker assignment template

Each assignment from the master must contain:

- ID and objective.
- Scope and owned files.
- Inputs and prerequisites.
- Required behavior or research questions.
- Deliverables.
- Acceptance criteria and appropriate verification.
- Resource limits and model/data access authorized by the task.
- Dependencies on other workers and required handoff information.

Worker completion reports must include:

- What was implemented or found.
- Files changed.
- Verification actually performed and its outcomes.
- Measured resource use, if applicable.
- Limitations, failures, and decisions needed from the master.

## 9. Current assignment status

### EVAL-001 — Workspace, benchmark system, and original-model baseline

Status: Assigned by the master at the user's request.

Objective: Build the evaluation workspace, download the original Qwen3-4B and benchmark data, and complete a small reproducible baseline run with an inspectable comparison report.

Ownership: The assigned worker owns implementation under `Work/gptcompress/`, project-local environment/dependency files, `gptcompress/models/`, `gptcompress/datasets/`, `gptcompress/experiments/`, and `gptcompress/reports/`. The master owns the root AGENTS.md and this assignment. Preserve existing user files. Inspect the current workspace before selecting package names and paths.

Authorization: Install required project-local dependencies, download public model checkpoints and benchmark datasets, and execute local evaluation. Use the existing Windows machine, RTX 3050 with 4 GB VRAM, and approximately 16 GB RAM. Report resource issues and recover through CPU staging, smaller batches, or checkpoint streaming. Paid compute and account setup require a separate user decision.

Required deliverables

1. Organize the method project with `Work/gptcompress/src/`, `Work/gptcompress/configs/`, `gptcompress/models/original/`, `gptcompress/models/compressed/`, `gptcompress/datasets/calibration/`, `gptcompress/datasets/benchmarks/`, `gptcompress/experiments/`, and `gptcompress/reports/`, adapting to existing content when needed.
2. Download and pin the original `Qwen/Qwen3-4B` high-precision checkpoint, tokenizer, and configuration. Record its immutable source revision and actual file sizes. Reuse existing caches where possible without duplicating large assets unnecessarily.
3. Implement a reproducible evaluation entry point using established scoring tools. Prefer EleutherAI lm-evaluation-harness for supported tasks and EvalPlus for coding. Verify current upstream interfaces and pin the versions actually used.
4. Provide benchmark configurations and dataset manifests for MMLU-Pro, GPQA Diamond, GSM8K, MATH-500, IFEval, BBH, HumanEval+/MBPP+, Belebele English/Arabic/French, and selected LongBench tasks. Download accessible data. Record gated or unavailable datasets with the exact access issue and continue independent work. Record licenses, revisions, splits, and checksums or dataset fingerprints.
5. Provide a development profile with fixed question IDs and a larger evaluation profile with disjoint held-out examples where feasible. Never use benchmark questions as compression calibration. Explicitly label custom subsets, protocol changes, and long-context restrictions.
6. Every run gets a unique directory, manifest, metrics, per-example predictions, and execution log. Preserve historical results. Include source revision or source-tree hash, parent model/run, compression configuration, dataset identity, question IDs, scoring protocol, prompts/chat-template identity, generation settings, seed, device placement, and software versions. Record errors and incomplete tasks explicitly.
7. Generate a local HTML report with benchmark scores, sample counts, available uncertainty estimates, checkpoint size, peak RAM/VRAM, measured timing, and comparisons against the original and parent variants. Include a quality-versus-size plot when multiple suitable runs exist and an individual-answer view. Show missing measurements explicitly. Use measured data in the initial report.
8. Complete the first original-model baseline in 16-bit precision using bounded CPU/GPU placement. Fix non-thinking mode and document decoding settings. Start with a real-model generation and memory check, then run a deterministic small subset of at least GSM8K and IFEval, targeting 10 examples each. Expand to other core tasks if resource use and runtime permit. This initial subset establishes that the complete workflow operates and provides timing evidence for scheduling the larger suite.
9. Document runnable commands for downloading assets, running each profile, evaluating a new model variant, and rebuilding the report. Include a factual completion report with exact commands, artifact locations, verified checks, and unfinished or access-blocked items.

Evaluation requirements

- Track every model variant through immutable artifacts and run metadata. Reuse baseline records when the evaluated model and protocol are identical.
- Use published scoring logic and preserve per-example data needed to reproduce aggregate scores. Label results according to the actual protocol.
- Record peak memory and timing using a documented measurement method. Distinguish estimates from measurements and weights-only disk size from runtime memory.
- Use an isolated execution environment for generated-code benchmarks. If suitable isolation is unavailable locally, prepare the coding configuration and data and report that execution dependency while continuing the other evaluations.
- Choose initial context and output limits conservatively for available memory. Record truncation and answer-length-limit failures. The initial run's time measurements should inform any larger run.
- Resume recoverable interrupted downloads and runs without overwriting completed experiments. Avoid loading duplicate full checkpoints.

Acceptance checks

- The pinned original checkpoint and accessible benchmark data are present and their manifests are readable.
- Evaluation runs through its documented entry point and produces real Qwen predictions, scores, and resource measurements for the initial subset.
- The generated report reads those artifacts and clearly labels sample counts and run status.
- Verify history preservation and comparison logic with focused checks. Use synthetic fixtures only for software verification and identify them as fixtures.
- Verify that calibration/development/held-out identities remain distinct, and that no missing benchmark is displayed as a zero score.
- Report any unmet criterion and its concrete cause. Implementation alone does not satisfy the initial-run requirement.

Worker communication: Send the master an early environment/disk/dependency assessment, report recoverable failures and decisions, and supply milestone updates during lengthy downloads or evaluation. The master reviews results and assigns subsequent compression experiments.

### ORG-001 — Group source entry points and inspector builds by project

Status: Complete, reviewed on 2026-11-22.

Objective: Reorganize and rename the mixed source entry points so compression, evaluation, and model inspection are clearly separated. Put the inspector executables under an inspector-specific distribution folder and give them descriptive names. Preserve behavior and all existing model, dataset, experiment, inspection, and report artifacts.

Ownership: The worker owns implementation moves/renames under `Work/src/`, import/reference updates in `Work/tests/`, `Work/README.md`, inspector build instructions, and relocation/renaming of generated inspector binaries under `Work/dist/`. The master owns this assignment. Do not alter compression algorithms, evaluation protocols, app behavior, benchmark data, model checkpoints, experiment outputs, inspection recordings/version registry, or unrelated user edits.

Target grouping (adjust exact module names only when required to preserve clear boundaries and imports):

- `src/gptcompress/compression/`: compression entry point and factorization implementation.
- `src/gptcompress/evaluation/`: asset preparation, evaluation entry point, experiment monitor, and report generation.
- `src/gptcompress/inspector/`: web and desktop entry points, inspection/recording/matrix/comparison modules, and static assets.
- Keep truly shared runtime modules in `src/gptcompress/` and do not create empty or duplicate compatibility layers.
- Put inspector binaries in `dist/model-inspector/`. Keep a clearly named canonical latest executable there; organize any retained sprint builds in a subordinate versions folder. Rename the files descriptively while preserving each retained binary's contents.

Required work

1. Map existing imports, dynamic paths, launch commands, PyInstaller inputs, test imports, and documentation references before moving files. Do not use CodeGraph unless an index exists at the repository root.
2. Move and rename `src/compress.py`, `src/evaluate.py`, `src/assets.py`, `src/progress.py`, `src/inspect_app.py`, and `src/inspect_desktop.py` into the right project packages with descriptive names. Relocate associated `gptcompress` modules and `inspector_static` assets so an inspector dependency is not left mixed with evaluation/compression modules.
3. Update package imports, static asset lookup, repository-root discovery, multiprocessing/job launch paths, process detection in the experiment monitor, tests' module references, README commands, report links, and PyInstaller commands for the new layout. Commands must work from the documented `Work` directory and must not depend on the caller's current directory unless that is stated.
4. Move the existing inspector executables into `dist/model-inspector/` and rename them consistently. Retain historical sprint executables where reports depend on them; keep the latest supported binary easy to identify. Update references without changing historical results or claiming an old binary was rebuilt.
5. Remove obsolete source entry points after all references have been migrated. Avoid leaving misleading stubs or duplicate copies.
6. Keep this as a structural refactor. Do not download models/data, run compression/evaluation jobs, change results, or add dependencies.

Acceptance

- Source files are grouped into the three project packages above, with only genuinely shared modules remaining at the package root.
- Entry-point and executable names communicate their purpose; build output is contained under `dist/model-inspector/`.
- A repository-wide static reference audit finds no stale paths for moved source files or renamed executables, except clearly labeled historical command records when necessary.
- Existing source and user data remain present; only intended source moves and inspector binary moves/renames occur.
- Report exact moves/renames, updated launch commands, and any references intentionally preserved for historical accuracy. Do not run the test suite or other tests in this assignment; the user requested organization and naming, not test execution.

Dependencies and limits: No package install, network access, large file read, model loading, test execution, inference, or experiment work is authorized. Coordinate through the master if an unrelated modified file, path conflict, or ambiguity about preserving a deliverable is found. Preserve all pre-existing user changes.

Completion record: Compression, evaluation, and inspector modules are grouped under their respective packages; current entry points and build references use the new names; the inspector executables are organized under `dist/model-inspector/`; stale bytecode for moved modules was removed. A static source/test reference audit found no stale imports or asset paths. Tests were not run, per assignment scope.
