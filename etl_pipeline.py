#!/usr/bin/env python3
"""
etl_pipeline.py — Extract / Transform / Load orchestrator for the daily
multi-market report (India · US · Europe).

Why
---
The legacy `daily_pipeline.sh` is a linear script that mixes fetching, scoring
and reporting. This wraps the same building blocks (market scans, the sentiment
pipeline, emerging-market fundamentals, the report builder) in an explicit ETL
shape with a curated data lake and per-run lineage, so each stage is
independently runnable, idempotent, and resumable.

Layout (all under ./data, LFS-tracked where it matters)
    data/raw/<date>/        EXTRACT  — raw pulls (universe snapshots, scan xlsx refs, news json)
    data/curated/           TRANSFORM— normalized parquet tables (fundamentals, darvas, sentiment, convergence, picks)
    data/reports/<date>/    LOAD     — email HTML + artifact HTML
    data/curated/etl_runs/  manifest — one JSON per run (stage status, timing, row counts, lineage)

Stages
    E1 extract_universe       validate/snapshot the ticker universes
    E2 extract_market_scans   ensure a fresh scan workbook exists per market (reuse today's, else note)
    E3 extract_news           pull market-mood + per-pick headlines via the RSS providers
    T1 transform_fundamentals scan workbook -> curated fundamentals parquet (+ PE zone)
    T2 transform_darvas       scan workbook -> curated darvas parquet (fresh-breakout flags)
    T3 transform_sentiment    news -> curated per-ticker sentiment parquet
    T4 transform_convergence  fundamentals x sentiment -> curated convergence parquet
    L1 load_report            build the email HTML + Artifact page (delegates to build_email.py)
    L2 load_manifest          write the run manifest (lineage + row counts + status)

CLI
    python3 etl_pipeline.py --phase all               # E -> T -> L
    python3 etl_pipeline.py --phase T,L               # skip extract, reuse today's raw
    python3 etl_pipeline.py --stage transform_darvas  # a single stage
    python3 etl_pipeline.py --markets US,EU --force    # ignore idempotency
    python3 etl_pipeline.py --dry-run                  # print the plan only
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/Downloads/code/python_files")


def _load_dotenv(path=None):
    """Minimal no-dependency .env loader. Reads KEY=VALUE lines from ./.env into
    os.environ WITHOUT overriding values already set in the environment
    (launchd/shell env wins). Quotes stripped; #comments and blanks ignored."""
    path = path or os.path.join(BASE, ".env")
    if not os.path.exists(path):
        return
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

RUN_DATE = date.today().isoformat()
RAW = os.path.join(BASE, "data", "raw", RUN_DATE)
CURATED = os.path.join(BASE, "data", "curated")
REPORTS = os.path.join(BASE, "data", "reports", RUN_DATE)
RUNS = os.path.join(CURATED, "etl_runs")
for d in (RAW, CURATED, REPORTS, RUNS):
    os.makedirs(d, exist_ok=True)

# Where each market's scan workbook lives (the EXTRACT product we curate from).
SCAN_GLOBS = {
    "US": "us_full_scan/us_full_scan_*.xlsx",
    "EU": "european_scan/european_market_scan_broad_*.xlsx",
    "IN": "indian_full_scan/indian_full_scan_*.xlsx",
}

# How to (re)build each market's scan when it's stale/missing (--refresh-scans).
# All run with cwd=BASE so their relative ./<market>_full_scan output dirs land
# where the ETL looks. EU is a local script; US/IN live in the GMS repo.
GMS_REPO = os.path.expanduser(os.environ.get("GMS_REPO", "~/repos/global-market-scanners"))
GMS_TOP = os.environ.get("GMS_TOP", "0")  # 0 = full universe
RESCAN_CMD = {
    "EU": [
        "full_european_market_scan.py",
        "--universe",
        "data/europe_broad_list.csv",
        "--label",
        "broad",
        "--batch",
        "120",
    ],
    "US": [os.path.join(GMS_REPO, "full_us_market_scan.py"), "--top", GMS_TOP],
    "IN": [os.path.join(GMS_REPO, "full_indian_market_scan.py"), "--top", GMS_TOP],
}
FUND_SHEET = {
    "US": "All_Fundamentals",
    "EU": "Fundamentals",
    "IN": None,
}  # IN scan carries no fundamentals


def _latest(patt):
    files = sorted(glob.glob(os.path.join(BASE, patt)), key=os.path.getmtime)
    return files[-1] if files else None


# ────────────────────────────── stage framework ──────────────────────────────
class Ctx:
    """Carries artifacts + lineage between stages within one run."""

    def __init__(self, markets, force, no_send=False, refresh_scans=False, hidden_gems=False):
        self.markets = markets
        self.force = force
        self.no_send = no_send
        self.refresh_scans = refresh_scans
        self.hidden_gems = hidden_gems
        self.artifacts = {}  # logical name -> path
        self.rows = {}  # logical name -> row count
        self.lineage = {}  # output -> [inputs]
        self.status = {}  # stage -> ok|skip|fail|dry
        self.timings = {}  # stage -> seconds
        self.notes = {}  # stage -> message

    def fresh(self, path):
        """Idempotency: an output is fresh if it exists and was written today."""
        if self.force or not path or not os.path.exists(path):
            return False
        return datetime.fromtimestamp(os.path.getmtime(path)).date().isoformat() == RUN_DATE


STAGES = []  # (name, phase, fn)


def stage(name, phase):
    def deco(fn):
        STAGES.append((name, phase, fn))
        return fn

    return deco


# ─────────────────────────────── EXTRACT ───────────────────────────────
@stage("extract_universe", "E")
def extract_universe(ctx):
    """Snapshot + sanity-check the ticker universes feeding the scans."""
    out = os.path.join(RAW, "universe_manifest.json")
    if ctx.fresh(out):
        ctx.artifacts["universe"] = out
        return "skip"
    manifest = {}
    for name, patt in [
        ("europe_broad", "data/europe_broad_list.csv"),
        ("global_master", "data/global_universe_master.csv"),
    ]:
        p = os.path.join(BASE, patt)
        if os.path.exists(p):
            n = sum(1 for _ in open(p)) - 1
            manifest[name] = {"path": patt, "tickers": n}
            ctx.rows[f"universe:{name}"] = n
    json.dump(manifest, open(out, "w"), indent=2)
    ctx.artifacts["universe"] = out
    ctx.lineage[out] = [m["path"] for m in manifest.values()]
    return "ok"


def _rescan(market):
    """Run a market's scan script (cwd=BASE) to refresh its workbook. Returns
    (ok, message). Skips gracefully if the script isn't present."""
    cmd = RESCAN_CMD.get(market)
    if not cmd:
        return False, "no rescan command"
    script = cmd[0] if os.path.isabs(cmd[0]) else os.path.join(BASE, cmd[0])
    if not os.path.exists(script):
        return (
            False,
            f"scan script not found ({os.path.relpath(script, BASE) if script.startswith(BASE) else script})",
        )
    r = subprocess.run([sys.executable, script, *cmd[1:]], cwd=BASE, capture_output=True, text=True)
    return (r.returncode == 0), (
        "rescanned"
        if r.returncode == 0
        else f"rescan failed: {(r.stderr or r.stdout).strip().splitlines()[-1][:80] if (r.stderr or r.stdout) else 'rc=%d' % r.returncode}"
    )


@stage("extract_market_scans", "E")
def extract_market_scans(ctx):
    """Locate each market's scan workbook (the raw OHLC+fundamentals extract).
    Reuse today's workbook if present; if stale/missing and --refresh-scans is
    set, trigger the market's scan script to rebuild it, then re-locate."""
    notes = []
    for m in ctx.markets:
        patt = SCAN_GLOBS.get(m)
        wb = _latest(patt) if patt else None
        stale = wb and datetime.fromtimestamp(os.path.getmtime(wb)).date().isoformat() != RUN_DATE
        if (not wb or stale) and ctx.refresh_scans:
            ok, msg = _rescan(m)
            notes.append(f"{m}: {'stale' if wb else 'missing'} → {msg}")
            if ok:
                wb = _latest(patt)  # pick up the freshly written workbook
        elif not wb:
            notes.append(f"{m}: NO scan workbook (run with --refresh-scans to build)")
        elif stale:
            notes.append(f"{m}: stale ({os.path.basename(wb)}) — --refresh-scans to rebuild")
        if wb:
            ctx.artifacts[f"scan:{m}"] = wb
    if notes:
        ctx.notes["extract_market_scans"] = notes
    return "ok"


@stage("extract_darvas_fragments", "E")
def extract_darvas_fragments(ctx):
    """Regenerate the IN/US Darvas HTML fragments the report embeds — after the
    scans are (re)fresh, so they reflect the latest workbooks."""
    made = []
    for m in ("IN", "US"):
        if m not in ctx.markets:
            continue
        dest = os.path.expanduser(f"~/Downloads/{m.lower()}_darvas.html")
        r = subprocess.run(
            [
                sys.executable,
                os.path.join(BASE, "darvas_breakouts.py"),
                "--market",
                m,
                "--top",
                "15",
            ],
            cwd=BASE,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            open(dest, "w").write(r.stdout)
            made.append(m)
    ctx.notes["extract_darvas_fragments"] = f"regenerated: {', '.join(made) or 'none'}"
    return "ok"


@stage("extract_news", "E")
def extract_news(ctx):
    """Pull market-mood headlines via the free RSS providers, land as json."""
    out = os.path.join(RAW, "news_mood.json")
    if ctx.fresh(out):
        ctx.artifacts["news"] = out
        return "skip"
    try:
        from sentiment_pipeline import SentimentPipeline

        sp = SentimentPipeline()
        mood = {m: (sp.get_market_mood(m) or {}) for m in ctx.markets}
    except Exception as e:  # pragma: no cover - provider/network variance
        ctx.notes["extract_news"] = f"sentiment unavailable: {str(e)[:80]}"
        mood = {}
    json.dump(mood, open(out, "w"), indent=2, default=str)
    ctx.artifacts["news"] = out
    for m, d in mood.items():
        ctx.rows[f"news:{m}"] = d.get("n_articles", 0)
    return "ok"


@stage("extract_ipo_tracker", "E")
def extract_ipo_tracker(ctx):
    """Detect IPOs / new listings and ADD them to the universe (ipo_tracker), so
    the reference + scans pick up post-IPO stocks. Runs before the exchange
    reference. Idempotent per day (skips if the IN snapshot was updated today);
    fully wrapped so a live-feed hiccup never breaks the pipeline."""
    snap = os.path.join(BASE, "data", "bhavcopy_cache", "universe_snapshots", "IN.json")
    if not ctx.force and ctx.fresh(snap):
        return "skip"
    try:
        import ipo_tracker

        found = ipo_tracker.track(ipo_tracker.DEFAULT_MARKETS, verbose=False)
        counts = {m: len(v.get("new", [])) for m, v in found.items()}
        ctx.rows["ipo_new"] = sum(counts.values())
        hit = ", ".join(f"{m}:{n}" for m, n in counts.items() if n)
        ctx.notes["extract_ipo_tracker"] = "new listings — " + (hit or "none")
    except Exception as e:
        ctx.notes["extract_ipo_tracker"] = f"failed: {str(e)[:90]}"
        return "fail"
    return "ok"


@stage("extract_exchange_reference", "E")
def extract_exchange_reference(ctx):
    """Build the all-exchange data-freshness reference + daily checklist (runs
    last in Extract, after any rescans) so stale market data is caught. Writes
    data/curated/exchange_reference.parquet + daily_checklist_<date>.md and
    flags any STALE exchanges in the run manifest."""
    try:
        import exchange_reference as xr

        df = xr.build_reference(date.fromisoformat(RUN_DATE))
        if df.empty:
            ctx.notes["extract_exchange_reference"] = "no universe files under data/global_universe"
            return "skip"
        pq = xr.write_parquet(df)
        md = os.path.join(CURATED, f"daily_checklist_{RUN_DATE}.md")
        open(md, "w").write(xr.checklist_md(df, date.fromisoformat(RUN_DATE)))
        ctx.artifacts["exchange_reference"] = pq
        ctx.artifacts["checklist"] = md
        ctx.rows["exchange_reference"] = len(df)
        stale = df[df["status"] == "STALE"]["exchange"].tolist()
        fresh = int((df["status"] == "FRESH").sum())
        ctx.notes["extract_exchange_reference"] = f"{len(df)} exchanges, {fresh} fresh" + (
            f" · ⚠ STALE: {', '.join(stale)}" if stale else " · none stale"
        )
    except Exception as e:
        ctx.notes["extract_exchange_reference"] = f"failed: {str(e)[:100]}"
        return "fail"
    return "ok"


# ─────────────────────────────── TRANSFORM ───────────────────────────────
def _pe_zone(pe, sm):
    if pd.isna(pe) or pe <= 0 or pd.isna(sm) or sm <= 0:
        return "n/a"
    r = pe / sm
    return "BUY" if r <= 0.8 else "FAIR" if r <= 1.1 else "CAUTION" if r <= 1.6 else "RICH"


@stage("transform_fundamentals", "T")
def transform_fundamentals(ctx):
    """Scan workbook fundamentals -> normalized parquet with a sector-relative PE zone."""
    frames = []
    for m in ctx.markets:
        wb = ctx.artifacts.get(f"scan:{m}")
        sheet = FUND_SHEET.get(m)
        if not wb or not sheet:
            continue
        df = pd.read_excel(wb, sheet_name=sheet)
        pe_col = "P_E_Ratio" if "P_E_Ratio" in df.columns else None
        if pe_col is None and "Earnings_Yield_%" in df.columns:
            df["P_E_Ratio"] = df["Earnings_Yield_%"].apply(
                lambda e: 100.0 / e if pd.notna(e) and e > 0 else np.nan
            )
            pe_col = "P_E_Ratio"
        df["P_E_Ratio"] = pd.to_numeric(df.get(pe_col), errors="coerce")
        med = df.groupby("Sector")["P_E_Ratio"].transform("median")
        df["PE_Zone"] = [_pe_zone(pe, sm) for pe, sm in zip(df["P_E_Ratio"], med)]
        df["Market"] = m
        keep = [
            c
            for c in [
                "Market",
                "Symbol",
                "Name",
                "Sector",
                "LTP",
                "P_E_Ratio",
                "PE_Zone",
                "Piotroski_Score",
                "CoffeeCan_Class",
                "Darvas_Signal",
            ]
            if c in df.columns
        ]
        frames.append(df[keep])
    if not frames:
        return "skip"
    out = os.path.join(CURATED, "fundamentals.parquet")
    allf = pd.concat(frames, ignore_index=True)
    allf.to_parquet(out, index=False)
    ctx.artifacts["fundamentals"] = out
    ctx.rows["fundamentals"] = len(allf)
    ctx.lineage[out] = [
        ctx.artifacts.get(f"scan:{m}") for m in ctx.markets if f"scan:{m}" in ctx.artifacts
    ]
    return "ok"


@stage("transform_darvas", "T")
def transform_darvas(ctx):
    """Scan workbook Darvas signals -> normalized parquet, flagging fresh breakouts."""
    frames = []
    for m in ctx.markets:
        wb = ctx.artifacts.get(f"scan:{m}")
        if not wb:
            continue
        try:
            df = pd.read_excel(wb, sheet_name="Darvas_Signals")
        except Exception:
            continue
        pos = pd.to_numeric(df.get("Position_in_Box%"), errors="coerce")
        chg = pd.to_numeric(df.get("Change%"), errors="coerce")
        df["fresh_breakout"] = (
            (df.get("Darvas_Signal") == "BREAKOUT_BUY") & pos.between(100, 120) & (chg > 0)
        )
        df["Market"] = m
        keep = [
            c
            for c in [
                "Market",
                "Symbol",
                "Name",
                "LTP",
                "Change%",
                "Darvas_Signal",
                "Box_Top",
                "Position_in_Box%",
                "fresh_breakout",
            ]
            if c in df.columns
        ]
        frames.append(df[keep])
    if not frames:
        return "skip"
    out = os.path.join(CURATED, "darvas.parquet")
    allf = pd.concat(frames, ignore_index=True)
    allf.to_parquet(out, index=False)
    ctx.artifacts["darvas"] = out
    ctx.rows["darvas"] = len(allf)
    ctx.rows["darvas:fresh"] = int(allf["fresh_breakout"].sum())
    return "ok"


@stage("transform_sentiment", "T")
def transform_sentiment(ctx):
    """Score per-pick news sentiment for the strongest fundamental names."""
    fund = ctx.artifacts.get("fundamentals")
    if not fund:
        return "skip"
    try:
        from sentiment_pipeline import SentimentPipeline

        sp = SentimentPipeline()
    except Exception as e:
        ctx.notes["transform_sentiment"] = f"unavailable: {str(e)[:80]}"
        return "fail"
    picks = pd.read_parquet(fund)
    rows = []
    for m in ctx.markets:
        sub = picks[
            (picks["Market"] == m)
            & (pd.to_numeric(picks.get("Piotroski_Score"), errors="coerce") >= 7)
        ]
        seen = set()
        for _, r in sub.head(40).iterrows():
            nm = str(r.get("Name") or r["Symbol"])
            if nm.upper() in seen:
                continue
            seen.add(nm.upper())
            s = (
                sp.get_ticker_sentiment(r["Symbol"], m, company_name=nm)
                if m == "EU"
                else sp.get_ticker_sentiment(r["Symbol"], m)
            )
            if s.n_articles > 0 and s.label != "NO_DATA":
                rows.append(
                    {
                        "Market": m,
                        "Symbol": r["Symbol"],
                        "Name": nm,
                        "sentiment": s.label,
                        "score": s.score,
                        "n_articles": s.n_articles,
                        "headline": (s.top_headlines[0]["title"] if s.top_headlines else ""),
                    }
                )
    out = os.path.join(CURATED, "sentiment.parquet")
    pd.DataFrame(rows).to_parquet(out, index=False)
    ctx.artifacts["sentiment"] = out
    ctx.rows["sentiment"] = len(rows)
    return "ok"


@stage("transform_convergence", "T")
def transform_convergence(ctx):
    """Join fundamentals (strong) x sentiment -> convergence table."""
    fund, sent = ctx.artifacts.get("fundamentals"), ctx.artifacts.get("sentiment")
    if not fund or not sent:
        return "skip"
    f = pd.read_parquet(fund)
    s = pd.read_parquet(sent)
    if s.empty:
        out = os.path.join(CURATED, "convergence.parquet")
        s.to_parquet(out, index=False)
        ctx.artifacts["convergence"] = out
        ctx.rows["convergence"] = 0
        return "ok"
    strong = f[pd.to_numeric(f.get("Piotroski_Score"), errors="coerce") >= 7][
        ["Market", "Symbol", "Sector", "PE_Zone"]
    ]
    conv = s.merge(strong, on=["Market", "Symbol"], how="inner")
    conv["signal"] = np.where(
        conv["sentiment"] == "POSITIVE",
        "HIGH_CONVICTION",
        np.where(conv["sentiment"] == "NEGATIVE", "CAUTION", "MIXED"),
    )
    out = os.path.join(CURATED, "convergence.parquet")
    conv.to_parquet(out, index=False)
    ctx.artifacts["convergence"] = out
    ctx.rows["convergence"] = len(conv)
    ctx.lineage[out] = [fund, sent]
    return "ok"


@stage("transform_hidden_gems", "T")
def transform_hidden_gems(ctx):
    """Full-universe only (--hidden-gems): the strongest picks the fast top-200
    morning scan can't reach — Piotroski>=8 across every market. Writes a ranked
    curated/hidden_gems_<date>.parquet (US also sorted smallest-first by
    Net_Profit_$M, so lesser-known names surface). Skipped on capped runs."""
    if not ctx.hidden_gems:
        return "skip"
    fund = ctx.artifacts.get("fundamentals")
    if not fund:
        return "skip"
    f = pd.read_parquet(fund)
    p = pd.to_numeric(f.get("Piotroski_Score"), errors="coerce")
    gems = f[p >= 8].copy()
    gems["Piotroski_Score"] = pd.to_numeric(gems["Piotroski_Score"], errors="coerce")
    if "Net_Profit_$M" in gems.columns:
        gems["_size"] = pd.to_numeric(gems["Net_Profit_$M"], errors="coerce")
        gems = gems.sort_values(["Piotroski_Score", "_size"], ascending=[False, True]).drop(
            columns="_size"
        )
    else:
        gems = gems.sort_values("Piotroski_Score", ascending=False)
    out = os.path.join(CURATED, f"hidden_gems_{RUN_DATE}.parquet")
    gems.to_parquet(out, index=False)
    ctx.artifacts["hidden_gems"] = out
    ctx.rows["hidden_gems"] = len(gems)
    ctx.lineage[out] = [fund]
    return "ok"


# ─────────────────────────────── LOAD ───────────────────────────────
@stage("load_report", "L")
def load_report(ctx):
    """Build the email HTML + Artifact page from the curated data (delegates to build_email.py)."""
    env = dict(os.environ)
    if ctx.hidden_gems:
        env["HIDDEN_GEMS"] = "1"  # build_email → distinct artifact file + title
    r = subprocess.run(
        [sys.executable, os.path.join(BASE, "build_email.py")],
        capture_output=True,
        text=True,
        cwd=BASE,
        env=env,
    )
    line = next((l for l in r.stdout.splitlines() if l.startswith("SUBJECT:")), "")
    ctx.notes["load_report"] = line or (r.stderr.splitlines()[-1] if r.stderr else "no output")
    # Hidden-gems runs write a SEPARATE artifact file so they publish to a
    # different Artifact URL than the fast morning report.
    art_name = (
        "daily_market_report_hiddengems_artifact.html"
        if ctx.hidden_gems
        else "daily_market_report_artifact.html"
    )
    for name, src in [
        ("email", "daily_market_report_%s.html" % RUN_DATE),
        ("artifact", art_name),
    ]:
        p = os.path.expanduser(f"~/Downloads/{src}")
        if os.path.exists(p):
            ctx.artifacts[name] = p
    return "ok" if r.returncode == 0 else "fail"


@stage("load_deliver", "L")
def load_deliver(ctx):
    """Email the report token-free via send_mailer.send (Gmail SMTP). Skips when
    --no-send is set or no GMAIL_USER/GMAIL_APP_PASSWORD is configured (report is
    still saved to ~/Downloads either way)."""
    report = ctx.artifacts.get("email")
    if not report or not os.path.exists(report):
        return "skip"
    if ctx.no_send:
        ctx.notes["load_deliver"] = "--no-send: report saved, not emailed"
        return "skip"
    pw = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not os.environ.get("GMAIL_USER") or len(pw) < 12 or "PUT-YOUR" in pw.upper():
        ctx.notes["load_deliver"] = (
            "no valid GMAIL_APP_PASSWORD (placeholder?) — report saved, not emailed"
        )
        return "skip"
    subject = ctx.notes.get("load_report", "").replace("SUBJECT: ", "").strip() or (
        f"Daily Market Report — {RUN_DATE}"
    )
    if ctx.hidden_gems:
        n = ctx.rows.get("hidden_gems", 0)
        subject = f"💎 Hidden Gems (full-market) — {n} strong picks · {subject}"
    try:
        import send_mailer

        ok = send_mailer.send(subject, "See the HTML report.", open(report).read())
        ctx.notes["load_deliver"] = (
            "emailed" if ok else "send returned False"
        ) + f": {subject[:60]}"
        return "ok" if ok else "fail"
    except Exception as e:
        ctx.notes["load_deliver"] = f"send failed: {str(e)[:80]}"
        return "fail"


@stage("load_manifest", "L")
def load_manifest(ctx):
    """Write the run manifest — status, timings, row counts, lineage."""
    out = os.path.join(RUNS, f"run_{RUN_DATE}.json")
    manifest = {
        "run_date": RUN_DATE,
        "generated": datetime.now().isoformat(),
        "markets": ctx.markets,
        "forced": ctx.force,
        "stage_status": ctx.status,
        "stage_seconds": {k: round(v, 2) for k, v in ctx.timings.items()},
        "row_counts": ctx.rows,
        "notes": ctx.notes,
        "artifacts": {k: os.path.relpath(v, BASE) for k, v in ctx.artifacts.items() if v},
        "lineage": {
            os.path.relpath(k, BASE): [os.path.relpath(i, BASE) for i in v if i]
            for k, v in ctx.lineage.items()
        },
    }
    json.dump(manifest, open(out, "w"), indent=2)
    ctx.artifacts["manifest"] = out
    return "ok"


# ─────────────────────────────── runner ───────────────────────────────
def run(
    markets, phases, only_stage, force, dry, no_send=False, refresh_scans=False, hidden_gems=False
):
    ctx = Ctx(markets, force, no_send=no_send, refresh_scans=refresh_scans, hidden_gems=hidden_gems)
    print(
        f"ETL run {RUN_DATE} | markets={','.join(markets)} | phases={','.join(phases)}"
        f"{' | STAGE='+only_stage if only_stage else ''}{' | FORCE' if force else ''}{' | DRY-RUN' if dry else ''}"
    )
    for name, phase, fn in STAGES:
        if only_stage and name != only_stage:
            continue
        if phase not in phases and not only_stage:
            continue
        if dry:
            print(f"  [{phase}] {name} … (dry)")
            ctx.status[name] = "dry"
            continue
        t0 = time.time()
        try:
            st = fn(ctx) or "ok"
        except Exception as e:
            st = "fail"
            ctx.notes[name] = f"EXC {type(e).__name__}: {str(e)[:120]}"
        ctx.timings[name] = time.time() - t0
        ctx.status[name] = st
        extra = ""
        {k: v for k, v in ctx.rows.items() if k == name or k.startswith(name.split("_")[-1])}
        if name in (
            "transform_fundamentals",
            "transform_darvas",
            "transform_sentiment",
            "transform_convergence",
        ):
            key = name.split("transform_")[-1]
            if key in ctx.rows:
                extra = f" · {ctx.rows[key]} rows"
        note = ctx.notes.get(name)
        note = ("; ".join(note) if isinstance(note, list) else note) if note else ""
        print(
            f"  [{phase}] {name:26} {st:5}{extra} ({ctx.timings[name]:.1f}s)"
            + (f"  ⚠ {note}" if note and st != "ok" else (f"  · {note}" if note else ""))
        )
    # always finalize the manifest unless we ran a single non-manifest stage
    if not dry and (not only_stage or only_stage == "load_manifest"):
        if "load_manifest" not in ctx.status:
            load_manifest(ctx)
    return ctx


def main():
    ap = argparse.ArgumentParser(description="ETL pipeline for the daily multi-market report")
    ap.add_argument("--markets", default="US,EU,IN", help="comma list: US,EU,IN")
    ap.add_argument("--phase", default="all", help="all or comma subset of E,T,L")
    ap.add_argument("--stage", default=None, help="run a single stage by name")
    ap.add_argument("--force", action="store_true", help="ignore idempotency / re-run")
    ap.add_argument(
        "--no-send", action="store_true", help="build/save the report but do not email it"
    )
    ap.add_argument(
        "--refresh-scans",
        action="store_true",
        help="rebuild a market's scan workbook when it's stale/missing",
    )
    ap.add_argument(
        "--hidden-gems",
        action="store_true",
        help="full-universe run: surface Piotroski>=8 picks beyond the top-200 (curated + subject)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    phases = (
        ["E", "T", "L"]
        if args.phase == "all"
        else [p.strip().upper() for p in args.phase.split(",")]
    )
    markets = [m.strip().upper() for m in args.markets.split(",")]
    ctx = run(
        markets,
        phases,
        args.stage,
        args.force,
        args.dry_run,
        no_send=args.no_send,
        refresh_scans=args.refresh_scans,
        hidden_gems=args.hidden_gems,
    )
    man = ctx.artifacts.get("manifest")
    if man:
        print(f"\nmanifest → {os.path.relpath(man, BASE)}")


if __name__ == "__main__":
    main()
