# Distributed Resource Manager (DRM)

Desktop application that plans, places and recovers GptCompress experiments across the local machine and cloud providers, under a hard experiment spending ceiling.

## Documentation

- Design specification: [docs/ORCHESTRATOR_ARCHITECTURE.md](docs/ORCHESTRATOR_ARCHITECTURE.md)
- Sprint roadmap: [docs/DELIVERY_PLAN.md](docs/DELIVERY_PLAN.md)
- Data and corpus rules it depends on: [../docs/DATA_ARCHITECTURE.md](../docs/DATA_ARCHITECTURE.md)

## Interfaces

- **`drm` CLI** — canonical machine-facing interface for AI agents and scripts: `--json` output with versioned schemas, stable exit codes, `--dry-run`, idempotency keys, `drm wait`, `drm explain` (spec Section 4)
- **Desktop app (PySide6)** — Providers, Data catalog, Experiment builder, Run monitor, Results screens
- **Local control daemon** — loopback-only JSON API + SQLite; both clients above talk to it

## Stack

Python 3.11+ · PySide6 · FastAPI (loopback only) · Typer · SQLite

## Status

Design agreed 2026-11-22; Sprints 0–1 implemented — contracts, SQLite store, loopback daemon, `drm` CLI, provider registry and artifact catalog with verified ingest/verify/pin/evict (87 passing tests). Sprint 2 — planner preview and local execution — next. Paid cloud launches remain disabled by default; nothing in this folder provisions, uploads, or spends.
