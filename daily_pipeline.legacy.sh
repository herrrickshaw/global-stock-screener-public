#!/bin/bash
# daily_pipeline.sh — full token-free Daily Market Brief.
# Runs entirely in local Python (no Claude / no LLM tokens) and emails the brief.
#
# Schedule it with cron or launchd (see daily_pipeline.plist). Set credentials
# in the environment first (GMAIL_USER, GMAIL_APP_PASSWORD, MAIL_TO).
#
#   ./daily_pipeline.sh            # refresh + screen + send
#   ./daily_pipeline.sh --draft    # refresh + screen + save brief_today.html
#   ./daily_pipeline.sh --gmail-draft  # refresh + screen + save into Gmail Drafts (IMAP)
#
# GMS_TOP controls how many symbols the global-market-scanners full-universe
# scans process (default 200, fast ~1min/market); set GMS_TOP=0 for the full
# universe (much slower -- hours, not minutes).
set -uo pipefail
cd "$(dirname "$0")"
PY=python3
GMS_REPO="$HOME/repos/global-market-scanners"
GMS_TOP="${GMS_TOP:-200}"
LOG="daily_pipeline_$(date +%Y%m%d).log"
{
  echo "=== Daily pipeline $(date) ==="
  echo "[1/7] India EOD refresh (official bhavcopy, incremental)"
  $PY bhavcopy_history.py 400 || echo "  bhavcopy refresh failed (will use cache)"
  echo "[2/7] India full screener scan"
  $PY scan_bhavcopy.py || echo "  scan failed (will use latest cache)"
  echo "[3/7] India + US combined report (fundamentals + street talk)"
  $PY daily_combined_report.py --market IN --html || echo "  IN combined report failed"
  $PY daily_combined_report.py --market US --html || echo "  US combined report failed"
  echo "[3b] refresh India CCC screen (screener.in)"
  $PY -c "import screener_in as s; s.ccc_screen().to_parquet('cache_seed/india_ccc_screen.parquet', index=False)" || echo "  CCC refresh skipped"
  echo "[4/7] AI-based hybrid screener (auto_screener.py) — India + US"
  $PY auto_screener.py --market IN --top 15 || echo "  IN auto-screener failed"
  $PY auto_screener.py --market US --top 15 || echo "  US auto-screener failed"
  echo "[5/7] global-market-scanners — full-universe US + India scan (GMS_TOP=$GMS_TOP)"
  if [ -d "$GMS_REPO" ]; then
    (cd "$GMS_REPO" && $PY full_us_market_scan.py --top "$GMS_TOP") || echo "  GMS US scan failed"
    (cd "$GMS_REPO" && $PY full_indian_market_scan.py --top "$GMS_TOP") || echo "  GMS India scan failed"
  else
    echo "  $GMS_REPO not found — skipping (clone github.com/herrrickshaw/global-market-scanners)"
  fi
  echo "[6/7] validate AI screener picks vs popular screens"
  $PY validation.py --market IN || echo "  IN validation failed"
  $PY validation.py --market US || echo "  US validation failed"
  echo "[7/7] build + send mailer"
  $PY send_mailer.py "$@"
  echo "=== done $(date) ==="
} >> "$LOG" 2>&1
echo "pipeline complete — see $LOG"
