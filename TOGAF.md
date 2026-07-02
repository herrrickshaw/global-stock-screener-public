# Enterprise Architecture — TOGAF® alignment

This project applies the **TOGAF Standard** (ADM cycle, Architecture Governance, and
Architecture Content) to the global stock screener. `governance.py` turns the
principles below into an **executable compliance check**.

> Educational/research system. TOGAF® is a standard of The Open Group.

---

## 1. Architecture Principles (register)

TOGAF principles are stated as **Name · Statement · Rationale · Implications**.

| # | Principle | Statement | Rationale | Implications / Control |
|---|-----------|-----------|-----------|------------------------|
| P1 | **Data is a shared asset** | One central store feeds every consumer. | Avoids duplicate, drifting copies. | Tiered LTM/STM + LMDB + serving views; `datalink` manifest is the single index. |
| P2 | **Single source of truth** | Each datum has exactly one authoritative producer. | Prevents contradictory data. | `DEPENDENCIES.md` §3 producer→consumer map; STM derived from LTM, not duplicated. |
| P3 | **Reuse before build (DRY)** | Common logic lives once. | Lowers cost & defects. | `frames.py`, `datalink`, `decision_engine` shared across modules. |
| P4 | **Data integrity is verifiable** | Every artifact is tamper-evident. | Trust in results. | `integrity.py` SHA-256 manifest + CI `verify-checksums` + branch protection. |
| P5 | **Security & privacy by default** | Secrets never enter code, chat, or VCS. | Protect credentials. | Creds via `.env` only (gitignored); `governance.py` scans for leaks. |
| P6 | **Everything is testable** | Logic is covered by fast, offline tests. | Safe change. | `tests/` (pure-logic), CI pytest matrix, `preflight.check`. |
| P7 | **Bounded cost / responsiveness** | Deep work is scoped to what clears filters. | Scales without waste. | CRUD `pipeline` promotes filter-clearers; memoised loads; incremental store. |
| P8 | **Graceful degradation** | Missing optional data never breaks a run. | Robust operation. | Source fallback chains (Trendlyne→screener→yfinance; yahoo→stooq). |
| P9 | **Explainability** | Every recommendation states its reasoning. | Human oversight. | `decision_engine` rationale + IMPAIRED flags; SHAP-style factor scores. |
| P10 | **Compliance over convenience** | No change lands unless governance passes. | Controlled evolution. | Protected `main` requires lint + tests + integrity to be green. |

---

## 2. ADM cycle → where it lives

| ADM phase | Intent | In this repo |
|-----------|--------|--------------|
| **Preliminary** | Principles, frameworks, tooling | This file; `pyproject`, pre-commit, CI |
| **A — Vision** | Scope, stakeholders, concerns | `README.md`, `DATA_SCHEMA_AND_ARCHITECTURE.md` §1 |
| **B — Business** | Capabilities, value | screening/decision capabilities; `AUTO_SCREENER.md` |
| **C — Data** | Data architecture | `DATA_SCHEMA_AND_ARCHITECTURE.md`, `MEMORY_ARCHITECTURE.md`, `DATA_SOURCES.md`, `DATA_LINKAGE.md` |
| **C — Application** | Application architecture | `DEPENDENCIES.md` (module + data graph) |
| **D — Technology** | Platform | LMDB, parquet, DuckDB (`query.py`), Git LFS |
| **E — Opportunities & Solutions** | Roadmap | `DATA_SOURCES.md` ranked opportunities |
| **F — Migration Planning** | Sequenced change | feature branches → PRs; `BRANCHES.md` |
| **G — Implementation Governance** | Conformance | CI gates, `integrity.py`, `governance.py`, branch protection |
| **H — Change Management** | Controlled evolution | pre-commit auto-refresh (dep-map, checksums); protected `main` |
| **Requirements Management** | Continuous | issues/PRs; `SECURITY.md`, `SCREENER_ACCESS.md` |

**Iteration & partitioning (TOGAF techniques):** the system is *partitioned* by market
(20) and tier (LTM/STM/serving/CDC) and delivered in *iterations* (feature branches),
matching TOGAF's Agile/Digital-enterprise guidance.

---

## 3. Architecture Content Framework (artifacts we hold)

| Artifact type | Deliverable |
|---------------|-------------|
| **Catalogs** | `company_list` (entities), `DATA_SOURCES.md` (sources), Architecture Repository (`cache_seed/`) |
| **Matrices** | `DEPENDENCIES.md` — module-impact matrix + data producer/consumer matrix |
| **Diagrams** | `DATA_SCHEMA_AND_ARCHITECTURE.md` (Mermaid: layered flow, sequence, ERD), `architecture.svg` |
| **Principles** | this register (§1) |
| **Building blocks** | ABBs = layers (ingestion/storage/serving/ML); SBBs = modules |

## 4. Enterprise Continuum / Architecture Repository

| Continuum stage | Here |
|-----------------|------|
| Foundation | frames, datalink, integrity, preflight (generic mechanics) |
| Common systems | market_memory, serving_layer, screener_kit, pipeline |
| Industry | strategies, screen_metrics, fundamental_metrics, validation (equity-screening) |
| Organization-specific | auto_screener, decision_engine, portfolio_evaluator, data-source auths |

**Repository landscape:** committed baselines on `main`; reference data in
`reference_seed/`; archived/experimental architectures on `archive/*` branches.

## 5. Architecture Governance (Phase G, executable)

```bash
python governance.py            # TOGAF compliance check — exit 1 if a control fails
```
Ties each principle to an enforceable control (integrity manifest, no secrets in VCS,
dependency map present, tests present, governance CI configured, docs present). Run in
CI/pre-commit so **non-conformances are caught before they land** — TOGAF's
"governance contributes to architecture development" made concrete.
