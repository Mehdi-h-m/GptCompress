# GptCompress worker requirements

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
- Default research constraint, inherited from the supplied report: no model training, fine-tuning, or distillation. Forward-pass calibration and analytic reconstruction are permitted. The user has not explicitly relaxed this constraint.
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

Ownership: The assigned worker owns implementation under `Work/`, project-local environment/dependency files, `models/`, `datasets/`, `experiments/`, and `reports/`. The master owns the root AGENTS.md and this assignment. Preserve existing user files. Inspect the current workspace before selecting package names and paths.

Authorization: Install required project-local dependencies, download public model checkpoints and benchmark datasets, and execute local evaluation. Use the existing Windows machine, RTX 3050 with 4 GB VRAM, and approximately 16 GB RAM. Report resource issues and recover through CPU staging, smaller batches, or checkpoint streaming. Paid compute and account setup require a separate user decision.

Required deliverables

1. Organize the workspace with `Work/src/`, `Work/configs/`, `models/original/`, `models/compressed/`, `datasets/calibration/`, `datasets/benchmarks/`, `experiments/`, and `reports/`, adapting to existing content when needed.
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
