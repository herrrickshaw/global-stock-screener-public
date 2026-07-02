# Agile operating model — SAFe® alignment

How this project runs on **Lean-Agile / SAFe** practices. `backlog.py` makes the
backlog + WSJF prioritisation + PI planning **executable**.

> Educational/research project run by a solo maintainer + AI pair; SAFe roles are
> mapped to that reality, not a large org.

---

## 1. Value stream & the "Agile Release Train"

**Value stream:** *raw exchange data → screening/ML → an explainable Buy/Hold/Sell
for an investor.* The **ART** (the continuously-delivering system) is the daily
pipeline: `daily_memory.sh` (bhavcopy → LTM/STM → serving/CDC → pipeline) plus the
CI/CD that ships features to `main`.

## 2. Backlog hierarchy (SAFe artifacts → here)

| SAFe artifact | Here |
|---------------|------|
| **Epic** | a theme (e.g. *Smart-money signals*, *Fundamentals depth*, *Data quality*) |
| **Capability / Feature** | a shippable module (e.g. `delivery_signal`, Trendlyne source) |
| **Story** | a commit within a feature branch |
| **Enabler** | infra work (integrity, dep-map, preflight, DuckDB, partitioning) |
| **Portfolio backlog** | `DATA_SOURCES.md` ranked opportunities |
| **Program backlog** | `backlog.json` (WSJF-ranked, via `backlog.py`) |

## 3. Prioritisation — WSJF (Weighted Shortest Job First)

```
WSJF = Cost of Delay / Job Size
     = (user-business value + time criticality + risk reduction) / job size
```
Highest WSJF first → maximise value delivered per unit effort. `backlog.py` computes
it; e.g. the 1-point "clamp absurd P/E" ranks above an 8-point signal build.

```bash
python backlog.py            # WSJF-ranked program backlog
python backlog.py --pi       # propose next Program Increment within a capacity budget
```

## 4. Cadence — PI, iterations, ceremonies (mapped to git)

| SAFe event | Here |
|------------|------|
| **PI Planning** | `backlog.py --pi` selects the next increment by WSJF + capacity |
| **Iteration / Sprint** | a feature branch delivered end-to-end |
| **Daily stand-up** | the analysis/status runs (`preflight`, `market_memory --status`) |
| **System demo** | the analysis outputs (scoreboard, screener results) |
| **Inspect & Adapt** | `DATA_SOURCES.md` opportunity re-ranking + backlog re-scoring |

## 5. Built-in quality & DevOps (CALMR)

- **Continuous Integration/Delivery:** every branch → PR → CI (lint, pytest matrix,
  integrity, governance) → merge to `main`.
- **Definition of Done (enforced, not aspirational):** a feature is *done* only when
  `governance.py` is **CONFORMANT** and all CI gates are green — i.e. tests pass,
  integrity manifest matches, dependency map current, no secrets in VCS, docs present.
- **Built-in quality:** pre-commit (ruff/black + dep-map + checksums); protected `main`.

## 6. Lean-Agile principles in practice

Take an economic view (**WSJF**), build incrementally with fast feedback (**feature
branches + CI**), base milestones on working systems (**every PR ships a runnable,
tested feature**), make value flow (**the daily ART**), and decentralise decisions
(**the CRUD pipeline promotes what clears filters, automatically**).

## Definition of Done checklist (per feature)
- [ ] Behaviour implemented + `--help`/docstring usage
- [ ] Offline unit tests added (pure logic)
- [ ] `ruff` + `black` clean; `DEPENDENCIES.md` regenerated
- [ ] `integrity.py --verify` OK; `governance.py` CONFORMANT
- [ ] Docs updated (README / relevant `*.md`)
- [ ] Merged via PR with all CI checks green
