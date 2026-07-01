# Branch map

`main` holds the **core global stock screener** (data pipeline, tiered memory,
serving layer, screening, ML, CRUD pipeline, integrity). Legacy/exploratory work is
preserved on dedicated branches so `main` stays lean.

| Branch | Purpose |
|--------|---------|
| `main` | core system — kept lean, PR-gated + integrity-verified |
| `archive/full-snapshot-2026-07` | complete pre-declutter snapshot (safety net) |
| `archive/legacy-scanners` | superseded per-market/daily scan + report scripts |
| `archive/expansion-screening` | universe-scaling / debt-expansion experiments |
| `archive/experiments` | unrelated research one-offs (sentiment, deoldify, etc.) |

Each archive branch is a full working copy **with the central LFS data**, so its
scripts still run against the same data. Nothing is deleted — only moved off `main`.
