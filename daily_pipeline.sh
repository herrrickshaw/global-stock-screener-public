#!/bin/bash
# daily_pipeline.sh — token-free Daily Market Report, orchestrated by the ETL pipeline.
# =====================================================================================
# Runs entirely in local Python (no Claude / no LLM tokens). The heavy fetching
# (India EOD, combined reports, European broad scan, Darvas fragments) is the
# EXTRACT layer; etl_pipeline.py then does the structured Extract(reuse) →
# Transform (curated parquet data lake) → Load (build report + email + manifest).
#
# Usage:
#   ./daily_pipeline.sh              # refresh + curate + build + EMAIL (needs Gmail creds)
#   ./daily_pipeline.sh --draft      # same, but save the report only (no email)
#
# Credentials (only needed to actually send) via environment:
#   GMAIL_USER, GMAIL_APP_PASSWORD, MAIL_TO   (Google App Password)
# Without them the report is still built + saved to ~/Downloads.
#
# Schedule: com.umashankar.dailybrief.plist (launchd, macOS) runs this daily.
# Legacy 7-step pipeline preserved as daily_pipeline.legacy.sh.
set -uo pipefail
cd "$(dirname "$0")"
PY=python3
GMS_REPO="$HOME/repos/global-market-scanners"
GMS_TOP="${GMS_TOP:-200}"
LOG="daily_pipeline_$(date +%Y%m%d).log"

# --draft (or --gmail-draft, --no-send) → build/save but do not email.
# --full / --hidden-gems (or env HIDDEN_GEMS=1) → full-universe hidden-gems run.
SEND_FLAG=""
GEM_FLAG=""
[ "${HIDDEN_GEMS:-0}" = "1" ] && GEM_FLAG="--hidden-gems"
for a in "$@"; do
  case "$a" in
    --draft|--gmail-draft|--no-send) SEND_FLAG="--no-send" ;;
    --full|--hidden-gems)            GEM_FLAG="--hidden-gems" ;;
  esac
done

{
  echo "=== Daily ETL pipeline $(date) ==="

  echo "[prep] India EOD refresh + IN/US combined reports (fundamentals + street talk)"
  $PY bhavcopy_history.py 400                       || echo "  bhavcopy refresh failed (using cache)"
  $PY scan_bhavcopy.py                              || echo "  India scan failed (using cache)"
  $PY daily_combined_report.py --market IN --html   || echo "  IN combined report failed"
  $PY daily_combined_report.py --market US --html   || echo "  US combined report failed"

  # The ETL now owns the market scans (EU local + US/IN via the GMS repo,
  # rebuilt when stale via --refresh-scans) and the Darvas fragments, then
  # curates → builds the report → emails (token-free) → writes the manifest.
  echo "[etl] Extract (auto-rescan stale) → Transform → Load (report + email + manifest)${GEM_FLAG:+  [hidden-gems]}"
  $PY etl_pipeline.py --phase all --markets US,EU,IN --refresh-scans $SEND_FLAG $GEM_FLAG \
                                                    || echo "  etl_pipeline failed"

  echo "=== done $(date) ==="
} >> "$LOG" 2>&1
echo "pipeline complete — see $LOG"
