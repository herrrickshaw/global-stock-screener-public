# Colab checkpoint notebooks

Runnable checkpoints for the system's big functions — each clones the repo, pulls the
**central LFS data** (5y LTM per market), and exercises one layer end-to-end.

| Notebook | Checkpoints |
|----------|-------------|
| [01_data_layer.ipynb](01_data_layer.ipynb) | LFS data → `market_memory` (LTM/STM), `datalink` (manifest/store), `serving_layer` (views + CDC), `export_company_list` |
| [02_screening.ipynb](02_screening.ipynb) | `screener_kit`/`strategies`, `screen_metrics`, fast serving screens, `validation` |
| [03_ml_and_crud.ipynb](03_ml_and_crud.ipynb) | `ml_supervised` (L1), `auto_screener` (L2), `rl_trader` (L3), `pipeline` (CRUD) |
| [04_portfolio_evaluator.ipynb](04_portfolio_evaluator.ipynb) | **upload holdings / tax-P&L → Buy/Hold/Sell by P/E** (`portfolio_evaluator`) |

Open any notebook and click the **Open in Colab** badge, then Runtime → Run all.
