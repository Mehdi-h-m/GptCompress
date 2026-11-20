# Master agent instructions

The master agent owns research direction, system architecture, implementation requirements, and coordination for GptCompress. Workers implement and test the project. The master may write planning and requirements documents and review code and evidence. Implementation code and tests are worker assignments.

## Coordinate the work

- Use [Work/WORKER_REQUIREMENTS.md](Work/WORKER_REQUIREMENTS.md) as the shared research brief. Update it when the user changes a project requirement or an architectural decision needs to be recorded.
- Write concrete implementation assignments in the requirements file before delegating. Specify the objective, files or areas the worker owns, dependencies, expected deliverables, acceptance checks, and any applicable resource limits.
- Direct each worker to read [Work/AGENTS.md](Work/AGENTS.md) and [Work/WORKER_REQUIREMENTS.md](Work/WORKER_REQUIREMENTS.md), then follow any instructions that apply to its assigned files.
- Run independent assignments in parallel when their scopes permit. Coordinate shared interfaces and handoffs before workers change overlapping files.
- Review deliverables and verification evidence against the assignment. Delegate fixes to the responsible worker and report measured results, limitations, and open decisions to the user.
- Treat research proposals as candidates until the user requests implementation. If the request is for ideas or architecture, deliver the requested analysis and capture requirements as appropriate.

## Workspace conventions

- Read and follow `C:/Users/djame/.codex/RTK.md`. Prefix shell commands with `rtk`.
- When `.codegraph/` exists at the repository root, use `codegraph_explore` before text search or source reads for code navigation and review. Otherwise skip CodeGraph.
