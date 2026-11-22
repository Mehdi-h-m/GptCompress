# Agent instructions

## Roles and coordination

- The coordinating assistant is the master agent. It produces plans, architecture, solution ideas, worker assignments, and implementation requirements. It reviews evidence and results. It does not write or edit implementation code or tests.
- Workers implement explicitly assigned tasks. Workers may write code and tests within their assigned scope.
- Before starting work, every worker must read `WORKER_REQUIREMENTS.md` and any instructions applicable to its assigned files.
- The master may write and maintain requirements and architecture documents directly.
- Do not interpret a research idea or a candidate approach as an instruction to implement it. An assignment must name the intended deliverable and acceptance criteria.
- Workers must preserve unrelated user changes and avoid editing files owned by another active worker. Report conflicts to the master.
- Routine, reversible work inside an assignment does not require repeated user confirmation. Surface material changes to research scope or assumptions to the master.

## Workspace layout

- `gptcompress/` — the compression method: source in `gptcompress/src/gptcompress/` (including the `gptcompress.inspector` package), tests, configs, reports, experiments, datasets, and models.
- `inspector/` — the VIS-001 desktop model inspector deliverable: tests, reports, inspection recordings, and dist builds.
- `drm/` — the Distributed Resource Manager orchestrator app; design in `drm/docs/ORCHESTRATOR_ARCHITECTURE.md`.
- `docs/` — cross-project architecture documents. `reports/` — generated dashboard only. Dot-prefixed directories are local runtime state and are not versioned.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- Use `codegraph_explore`, passing the project path, or `codegraph explore "<symbol names or question>"`.
- Name a file or symbol to inspect its current source. Do not re-read source already supplied by CodeGraph without a reason.
- If there is no `.codegraph/` directory, skip CodeGraph entirely. Indexing is the user's decision.
<!-- CODEGRAPH_END -->

## Shell commands

Read and follow `C:/Users/djame/.codex/RTK.md`. Prefix shell commands with `rtk`; use `rtk proxy` where appropriate.
