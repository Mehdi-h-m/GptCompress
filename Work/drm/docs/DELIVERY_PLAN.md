# DRM delivery plan

Sprint roadmap for implementing the [Distributed Resource Manager](ORCHESTRATOR_ARCHITECTURE.md) · 2026-11-22

This document is a master-agent roadmap. It assigns no code and records no runs. Each sprint is dispatched as a worker assignment — deliverables and acceptance criteria written into `drm/` — before its work starts, per `WORKER_REQUIREMENTS.md`. Implementation lives in `drm/src/drm/` and `drm/tests/`; the venv gets an editable install of `drm/src` in Sprint 0 so `drm` is importable from day one.

## Slice mapping

Spec Section 10 defines six delivery slices. Sprints close them in this order:

| Sprint | Closes | Slice |
| --- | --- | --- |
| 0 | — | Scaffold, contracts, daemon and CLI skeletons |
| 1–3 | Slice 1 | Local control plane and catalog |
| 4–5 | Slice 2 | One free cloud route |
| 6 | Slice 3 | Automated placement |
| 7 | Slice 4 | Safe parallel execution |
| 8 | Slice 5 | Cost controls |
| 9 | Slice 6 | Real Qwen3 smoke run |
| 10 | — | Full desktop parity (spec Section 2) |
| 11 | — | Modal adapter (spec Section 10 note) |

## Target package layout (fixed in Sprint 0)

```
drm/
  src/drm/
    contracts/      JSON envelope, error records, state enums, task-manifest schema
    store/          SQLite schema, migrations, transactional repositories
    daemon/         FastAPI loopback API, scheduling loop, reconcile-on-start
    catalog/        logical and physical artifacts, verification, eviction
    providers/      adapter contract and registry; local, hf, kaggle, modal, mock
    planner/        DAG, candidate generation, selection, decision records
    ledger/         quotes, reservations, settlements, credits
    workers/        worker runtime and versioned task implementations
    cli/            Typer application, envelopes, exit codes
    ui/             PySide6 shell and screens
  tests/            unit and fault-injection suites
  docs/             the design specification and this plan
```

## Sprints

### Sprint 0 — Scaffold, contracts, daemon and CLI skeletons

**Goal.** A runnable skeleton whose interfaces every later sprint builds on, so contract mistakes surface before any provider work.

**Deliverables.**
- Package scaffold per the layout above; `pyproject.toml` with `drm` (and optional `drmgr`) console entry points; editable install into `.venv`.
- Contracts module: versioned envelope `{"contract": 1, ...}`, JSON error-record schema, job and operation state enums (`queued`…`cancelled`, plus `awaiting_user` and `reconciling`), task-manifest schema v1 stub.
- Store: SQLite schema v1 created once through a migration framework — `providers`, `artifacts`, `artifact_copies`, `jobs`, `attempts`, `operations`, `decisions`, `ledger_entries`, `reservations`, `audit_log` (columns for later sprints included from the start).
- Daemon: FastAPI app bound to `127.0.0.1` with `/health` and `/status`; scheduling loop stub that pauses when the coordinator is offline.
- CLI: Typer app with global `--json`, exit-code mapping for 0/1 wired, `drm status`, `drm daemon start|stop`.
- Provider adapter interfaces (`discover`, `probe`, `quote`, `submit`, `status`, `cancel`, `list_usage`; storage `head`/`read`/`write`/`list`/`delete`) and a mock-provider framework with failure injection used by all later fault tests.

**Acceptance.** `drm daemon start` then `drm status --json` exits 0 with a contract-versioned document; human mode prints a table; with the daemon stopped the CLI exits 1 with a JSON error record; the store rolls back a forced failed transaction; `python -m unittest discover -s drm/tests` is green. Refs §3, §4.

### Sprint 1 — Provider registry and artifact catalog

**Goal.** Providers and data become first-class records with verified copies, surviving restarts.

**Deliverables.**
- Registry separating adapter type from instances; connect/list/pause for a `local-runner` instance and two `filesystem-store` instances; credential-reference field present in the schema.
- Catalog: logical artifacts (ID, content hash, type, byte count, parents, split identity) and physical copies (provider, locator, immutable version, checksum, status `durable`/`staged`/`rebuildable`, access class, retention policy).
- Ingest, pin, evict and verify flows over filesystem stores: chunked SHA-256, atomic publish, verify-on-read, eviction gated on references and active leases.
- CLI: `drm providers connect|list|probe|pause`, `drm data ingest|list|pin|evict|verify`; `--dry-run` on mutating commands; idempotency keys accepted and stored.
- Restart recovery: a daemon killed mid-ingest reconciles the operation to a terminal state on restart with the catalog consistent.

**Acceptance.** Register one local runner and two storage locations; ingest a fixture with `--verify`; flip a byte in a copy and verification fails with exit 1 and a JSON error; restart the daemon and recover state; the whole flow repeats through `drm` with JSON output and stable exit codes. Refs §3, §6.1–6.2, §9.

### Sprint 2 — Planner preview and local execution

**Goal.** End-to-end local execution: plan, decide, run one bounded task, explain the decision.

**Deliverables.**
- Job model and trivial single-job DAG: dependencies, resource floor and estimated peak, max attempts, timeout, output format.
- Planner v0: candidates filtered by hard constraints; selection records decision inputs and alternatives with a deterministic tie-break.
- Local runner adapter: probe → stage verified inputs from the catalog → execute the task manifest → result manifest → validate → publish to catalog.
- CLI: `drm plan preview <experiment>` (placement, estimated cost range, blockers), `drm run submit|status|cancel`, `drm wait <op> --timeout`, `drm jobs retry <id>`, `drm explain <decision>`; exit code 2 on validation or policy blocks.
- Restart mid-run: a daemon killed while a task runs reconciles the attempt on restart — re-run or adopt the result — never two accepted outputs.

**Acceptance.** Preview → submit → wait → a verified artifact lands in the catalog, entirely through `drm`; `drm explain` replays the placement decision; restart mid-run recovers without duplicate output. Refs §4, §7.3, §9.

### Sprint 3 — Desktop shell (closes Slice 1)

**Goal.** The GUI proves parity with the CLI over the same daemon.

**Deliverables.**
- PySide6 app as a daemon client: Providers, Data catalog and Run monitor screens only.
- The same operations through the GUI: connect a provider, ingest and verify an artifact, preview a plan, submit, watch to terminal state; blockers and decision evidence displayed.
- Start/stop the daemon from the app; connection status indicator.

**Acceptance.** The Sprint 1–2 flow performed once in the GUI and once via `drm` ends in an identical state — Slice 1 evidence is complete. Refs §2, §3.

### Sprint 4 — Hugging Face durable storage

**Goal.** The durable private store from the data architecture, reachable through the same catalog.

**Deliverables.**
- HF Hub storage adapter: `head`/`read`/`write`/`list`/`delete` over private dataset and model repos; ranged and resumable downloads; checksum verification.
- Credentials referenced by ID, held in a credential store; never in manifests, logs or notebook outputs.
- `hf://datasets/...` ingest: pinned public upstream fetch and private release push; per-connection usage observation recorded.
- Two instances of the HF adapter selectable independently.
- CLI and desktop: connect/probe/refresh, ingest from and publish to HF, list copies showing `durable` state.

**Acceptance.** Ingest a pinned public corpus shard with verification; publish a private release to the user's HF account and read it back verified; probe reports documented storage limits and observed usage; zero spend. Refs §5 (HF profile), §6.

### Sprint 5 — Kaggle assisted execution (closes Slice 2)

**Goal.** First remote compute route, assisted, with durable-checkpoint recovery.

**Deliverables.**
- Kaggle assisted notebook adapter: prepare a self-contained task bundle (manifest, pinned inputs, worker code) and emit exact notebook launch instructions; `awaiting_user` state and exit code 3.
- Result import: pull declared notebook output files only, validate against the result manifest (IDs, checksums, numerical profile), accept into the catalog.
- Durable checkpointing to HF storage; on a lost or expired session the coordinator re-dispatches only the missing work and reads completed results from storage.
- Reconcile path: ambiguous outcome → `reconciling` state, exit code 4; provider state checked before any duplicate submission.
- Probe records GPU type, session duration, quota and output-persistence behavior before any job shape is scheduled.

**Acceptance.** Connect HF private storage and Kaggle assisted execution; submit one bounded private task bundle; import its validated remote output; simulate a missing scratch session and recover using the durable checkpoint. Refs §5 (Kaggle), §8 example, §9.

### Sprint 6 — Automated placement (closes Slice 3)

**Goal.** The planner picks placements from evidence, and accounts stop being special cases.

**Deliverables.**
- Candidate generation: usable replicas, staging routes, output destinations per eligible placement; wall-time and cash estimates with uncertainty; free-quota models quoted as zeros with documented quota bounds.
- Selection policy: hard-constraint filter → lower estimated cash → earlier completion → fewer bytes moved; deadline preference; alternatives recorded for `drm explain`.
- Allowance pools: quota and credits tracked at account or billing scope; connections sharing an allowance reserve against one pool; distinct allowances stay separate; shared limits counted once.
- Stale quotes and unverified routes blocked from auto-selection until refreshed.
- Re-plan triggers: probe changes, stalled jobs, invalid copies; unfinished work re-planned from completed outputs.
- Performance bootstrap: bounded benchmarks recording throughput, setup, peak memory and checkpoint timing per hardware × workload profile into an observation store.

**Acceptance.** Add a second compatible backend and two accounts on one adapter without changing workflow definitions; both accounts are independently selectable, distinct allowances tracked separately, shared limits counted once; placement changes when memory, cache location, cost or quota differ. Refs §5, §7.

### Sprint 7 — Safe parallel execution (closes Slice 4)

**Goal.** Concurrent shards across backends without corruption or double work.

**Deliverables.**
- Shard partitioning and dispatch: unique work-unit IDs, independent shards concurrent across eligible backends, session-sharing rules that amortize model loading.
- Result validation and merge: exact shard IDs, merge against a same-profile reference; incompatible numerical profiles never pooled; cross-profile pooling gated on a small comparison with declared tolerances.
- At-least-once execution with exactly one accepted output per logical shard: coordinator transaction on the acceptance path; duplicates land in diagnostics.
- Fault injection through mock providers: interruption mid-attempt, duplicate result submissions, corrupt transfer bytes.

**Acceptance.** Two disjoint shards execute across eligible backends and merge to match a same-profile reference; injected interruption, duplicates and corrupt transfers are handled; resume touches only unfinished shards. Refs §7.4–7.5, §8, §9.

### Sprint 8 — Cost ledger and spending controls (closes Slice 5)

**Goal.** The ceiling becomes enforceable, not advisory.

**Deliverables.**
- Ledger: quoted, reserved worst-case, provisional, settled, provider-scoped expiring credits, free-quota consumption, recurring obligations; settled charges and active reservations never double-counted.
- Transactional reservation before submit, held until cancellation, settlement or reconciliation; ceiling preserved across calendar and billing cycles and app restarts.
- Paid-launch gate: paid work auto-launches only when enabled for the run, charges are enforceably bounded, and the reservation fits the ceiling; unknown or unbounded charges exclude the route; standing storage gets a retention window and reservation.
- `drm budget set`, remaining-liability reporting, in-run budget changes within the ceiling.
- Simulated fault tests with mock providers: credit expiry, stale quotes, late charges, concurrent reservations, a provider that cannot bound storage costs.

**Acceptance.** The planned $10 cap survives an app restart under the fault suite; unsafe paid routes remain unlaunched. Refs §9.

### Sprint 9 — Qwen3 task pack and real smoke run (closes Slice 6)

**Goal.** The actual GptCompress pipeline runs through DRM on real checkpoints.

**Deliverables.**
- Versioned task types: calibration capture (shard sums `G_s = X_s X_s^T` plus valid-token counts, per-layer-group memory budget declared), reducer (fixed shard order, schema and arithmetic validation, exact input coverage), factorization publish (factors under original tensor identifiers, save/reload verification), bounded teacher-forcing and recovery, evaluation diagnostics.
- Local end-to-end smoke: representative activation statistics → factorize a bounded subset of real matrices → verified factor artifact → matched diagnostics → a small recovery task.
- All artifacts in the catalog, all decisions in the ledger, all runs through the CLI and desktop.

**Acceptance.** Documented memory, throughput, storage and costs for the bounded smoke run before any scaling decision. Refs §6 keep-list, §8.

### Sprint 10 — Full desktop parity

**Goal.** Every screen in spec Section 2 exists with a CLI equivalent.

**Deliverables.**
- Experiment builder: teacher, source release, workflow, quality profile and budget; preview with dependencies, placement, transfer volumes, duration range, cost range and blockers; manual notebook actions shown with exact run and shard until the provider confirms.
- Results: matched-protocol variant comparison, provider provenance per output, export of manifests, metrics and the itemized ledger.
- Run monitor upgrades: live attempts, resource peaks, reserved versus actual spend, planned moves, pause/resume/cancel and in-run budget change.

**Acceptance.** A spot-check matrix shows every Section 2 screen action has a CLI equivalent; the Sprint 9 flow is drivable entirely from the GUI.

### Sprint 11 — Modal adapter (optional extension)

**Goal.** Automated higher-memory batches once the free route works.

**Deliverables.**
- Modal adapter: managed batch execution with bounded duration and concurrency; startup, CPU, RAM and storage inside quotes; provider-side retry reconciliation; workspace spending limit enforcement.

**Acceptance.** A bounded higher-memory batch runs with reservation and settlement in the ledger; preemption and retries reconcile without duplicate outputs. Gate: only after Sprints 5 and 8.

## Deferred

Colab assisted runs, R2 or other object stores, GCP profiles, and the Lightning/Runpod/Vast candidates stay future extensions (Section 5 lists them as candidates). No sprint in this plan touches them.

## Sequencing and standing constraints

- Sprints run in order 0→11; Sprint 10 and Sprint 11 touch mostly disjoint areas (`ui/` vs `providers/`) and may run as parallel assignments after Sprint 9 if ownership scopes stay separate.
- Free tiers only through Sprint 7. Sprint 8's fault tests are simulated. Nothing launches paid work before the gate in Sprint 8, and paid launches stay disabled by default until explicitly enabled for a run.
- Real HF and Kaggle accounts are touched from Sprint 4 onward: read-only public fetches first; private writes only to the user's own repos; zero spend throughout.
- Mock providers and injected failures come before real accounts in every slice (Section 10).
- Each sprint ships its tests with the implementation; no sprint is accepted without its acceptance evidence above.
- The master writes the worker assignment for each sprint before dispatch and reviews acceptance evidence before closing it.
