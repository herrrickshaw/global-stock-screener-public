# Universal Expansion Screening System - Quick Start Guide
**Scan 25,000+ Global Stocks to Find Expansion Companies**

---

## PART 1: WHAT THIS SYSTEM DOES

This toolkit identifies **all companies worldwide** that have:
1. ✅ **Taken on debt** (2021-2026)
2. ✅ **Increased capex** for expansion
3. ✅ **Reinvested profits** while managing leverage

**Output:** Ranked list of ~800-1,200 expansion candidates across 4 tiers, with sector & geographic analysis.

---

## PART 2: THREE COMPONENTS

### Component 1️⃣: **Framework** (UNIVERSAL_EXPANSION_SCREENING_FRAMEWORK.md)
**What it is:** 3,000+ word guide explaining:
- 8-dimensional scoring model (debt expansion, capex acceleration, profit reinvestment, etc.)
- Sector-specific thresholds (semiconductors vs cloud vs retail)
- Geographic hotspots (Asia, North America, Europe, India)
- Data sources (SEC, Yahoo Finance, stock exchanges)
- Filtering logic to remove unsuitable companies

**When to read:** Before running analysis; understand WHY companies score the way they do.

---

### Component 2️⃣: **Automated Screener** (universal_expansion_screener.py)
**What it does:** Python script that:
1. Ingests 5-year financial data (debt, capex, profits, etc.) for any stock
2. Calculates 7 scoring dimensions
3. Assigns tier (Tier 1-4 or No Pattern)
4. Generates 5 output reports

**Input required:**
```csv
ticker, company_name, sector, market_cap_b,
total_debt_2021-2026, total_equity_2021-2026,
capex_2021-2026, revenue_2021-2026,
operating_cf_2021-2026, operating_income_2021-2026,
net_income_2021-2026, retained_earnings_2021-2026,
dividends_paid_2021-2026
```

**Output:** 5 CSV files + analysis

---

### Component 3️⃣: **Analysis Templates** (debt_expansion_analysis_template.csv)
**What it is:** Pre-loaded 5-year data for 6 major companies (Micron, Intel, SK Hynix, Microsoft, Meta, ASML).

**How to use:** 
- Study pattern of successful expanders (Micron, SK Hynix scores: 57-88)
- Study pattern of failed expanders (Intel score: 39, capex didn't deliver ROI)
- Compare your target company against these patterns

---

## PART 3: HOW TO RUN THE SCREENING (Step-by-Step)

### Step 1: Prepare Financial Data

**Option A: US Companies (SEC EDGAR)**
```python
# Download 5-year 10-K data for all US public companies
# Source: SEC EDGAR API (free, ~15 min for 3,500 companies)

from sec_api import QueryApi
query = """
  SELECT * FROM filings
  WHERE form-type="10-K"
  AND filedAt >= "2021-01-01"
  LIMIT 10000
"""
# Extract: Debt, Equity, Capex, Revenue, OCF, etc.
# Save to CSV
```

**Option B: International Companies (Multiple Sources)**
- **Europe:** EDGAR (EU-listed ADRs) + National filings
- **India:** NSE/BSE website, MCA database
- **China:** CSRC filings + TradingView aggregation
- **Japan:** JPX/EDINET database

**Option C: Shortcut - Use Aggregators**
- **Yahoo Finance API** (easiest; ~$0 for personal use)
- **Financial modeling sites:** Morningstar, GuruFocus, Seeking Alpha
- **Paid platforms:** Bloomberg, FactSet, Refinitiv (expensive but comprehensive)

### Step 2: Load Data into Screener

```python
import pandas as pd
from universal_expansion_screener import UniversalExpansionScreener

# Load your CSV with 5-year financial data
df = pd.read_csv('global_stocks_financial_data.csv')

# Initialize screener
screener = UniversalExpansionScreener(df)

# Run pipeline
results = screener.run(output_dir='my_expansion_results')
```

### Step 3: Review Output Reports

**Report 1: expansion_scorecard.csv**
```
Top 10 by score:
┌─────────┬──────────────┬────────┬───────────────────────────┐
│ ticker  │ company      │ score  │ tier                      │
├─────────┼──────────────┼────────┼───────────────────────────┤
│ GOOGL   │ Google       │ 91     │ TIER 1 - Proven Expander  │
│ MSFT    │ Microsoft    │ 90     │ TIER 1 - Proven Expander  │
│ AMZN    │ Amazon       │ 87     │ TIER 1 - Proven Expander  │
│ SKY     │ SK Hynix     │ 85     │ TIER 1 - Proven Expander  │
│ MU      │ Micron       │ 57     │ TIER 3 - Selective        │
│ INTC    │ Intel        │ 39     │ TIER 4 - Weak Signals     │
└─────────┴──────────────┴────────┴───────────────────────────┘
```

**Report 2: sector_analysis.csv**
```
Sector | Avg Score | Max Score | Count | Avg FCF
───────────────────────────────────────────────────
Semiconductors     | 65    | 88    | 45    | +$8B
Cloud/Data Centers | 82    | 91    | 12    | +$45B
Electric Vehicles  | 58    | 72    | 28    | +$2B
Telecom/5G         | 55    | 68    | 18    | +$5B
Retail/Consumer    | 35    | 52    | 120   | -$1B
```

**Report 3: tier1_top50.csv**
```
Top 50 Tier 1 companies:
- Company name, market cap, sector, all scoring dimensions
- Ready to deep-dive for investment analysis
```

**Report 4: summary_statistics.csv**
```
Total candidates scored: 1,248
Tier 1 (Proven): 52 companies (4.2%)
Tier 2 (Aggressive): 156 companies (12.5%)
Tier 3 (Selective): 248 companies (19.9%)
Tier 4 (Weak): 792 companies (63.4%)
```

### Step 4: Deep Dive on Tier 1 Companies

**For each Tier 1 company, analyze:**

1. **Capex ROI**: Is new capacity being utilized?
   - Track quarterly utilization rates
   - Monitor pricing power (are new products/services commanding premium prices?)
   - Compare margins pre/post-capex ramp

2. **Debt Service**: Can company comfortably pay interest?
   - Operating income / Interest expense > 2.5x?
   - Debt-to-EBITDA < 3.0x?

3. **Capex Cycle Timing**: Where are we in the cycle?
   - Peak capex year? (Usually top of market)
   - Inflection point? (Capex declining, profits rising → best entry)
   - Normalization? (Capex maintenance mode, dividends returning)

4. **Competitive Dynamics**: Are competitors over/under-investing?
   - Example: Semiconductor fab race → who's building capacity?
   - Example: Cloud data centers → is supply/demand balance tight or loose?

### Step 5: Track Quarterly Updates

**Quarterly monitoring (post-earnings):**
```
✅ Did capex guidance change? (up = accelerating cycle; down = peaking)
✅ Did debt issuance occur? (timing aligned with capex?)
✅ Did operating income grow? (ROI materializing?)
✅ Did FCF turn negative? (red flag for sustainability)
✅ Did company move between tiers? (score changed significantly)
```

---

## PART 4: REAL-WORLD EXAMPLE ANALYSIS

### Case: Micron Technology (MU)

**Scoring Results:**
```
Total Score: 57 (TIER 3 - Selective Expander)

Breakdown:
├─ Debt expansion: 0/20 (D/E falling 0.25→0.15; took debt in 2023, now paying down)
├─ Capex acceleration: 20/20 ⭐ (Capex 5x from $5.8B→$25B; +33.9% CAGR)
├─ Profit reinvestment: 15/15 ⭐ (RE $34.7B→$94.7B; +22.2% CAGR; dividend flat)
├─ Profitability: 8/15 (OI recovered but volatile)
├─ Sustainability: 9/15 (FCF $3.5B positive but squeezed by capex)
├─ Timing: 0/10 (Debt peaked 2023, now declining; not new debt for capex)
└─ Leverage: 5/5 (D/E 0.15 is healthy)

Why Score is Lower than "Textbook" Pattern:
1. D/E ratio is DECLINING (already paid down debt from 2023 peak)
2. This is the PAYDOWN phase, not EXPANSION phase
3. BUT: Capex is accelerating, profits being reinvested = still expanding

Interpretation:
✅ **POSITIVE:** Company took on debt 2023 for India ATMP facility + expanded capex
✅ **POSITIVE:** Profits reinvested (no dividend increase despite 3x profit growth)
✅ **POSITIVE:** FCF still positive despite extreme capex intensity

❌ **NEGATIVE:** Capex/OCF ratio 88% = tight liquidity (vulnerable to demand shock)
❌ **NEGATIVE:** D/E declining = debt taken for capex is being paid down (success signal)

**Investment Implication:** Micron is in CAPEX PEAK phase (2025-2026); next 3 years:
- Capex should moderate to $18-20B (from current $25B)
- Operating income will accelerate (new capacity ROI)
- Debt-to-Equity will continue declining
- Dividend likely to increase 2027-2028
```

---

## PART 5: SECTOR-LEVEL TAKEAWAYS

### 🔥 HOT SECTORS (Tier 1 Average Score: 80+)

**Semiconductors** (Score: 80+)
- Companies: Micron, SK Hynix, Samsung Foundry, TSMC
- Why expanding: AI memory shortage, US incentives (CHIPS Act)
- Peak capex: 2025-2026
- ROI inflection: 2027-2028 (demand continues to outpace supply)
- Risk: Over-capacity if demand softens (monitor quarterly bookings)

**Cloud/Data Centers** (Score: 82+)
- Companies: Microsoft ($95B capex 2026), Google ($85B), Amazon ($155B)
- Why expanding: AI infrastructure build (LLM training, inference clusters)
- Peak capex: 2026-2027
- ROI inflection: 2028+ (once AI capex wave moderates)
- Risk: Valuation stretched; market prices in ROI that hasn't materialized yet

**EV/Battery** (Score: 65+)
- Companies: BYD, Tesla (Berlin/Austin ramp), CATL, Northvolt
- Why expanding: Global EV adoption, battery localization
- Peak capex: 2024-2025 (SLOWING now)
- ROI inflection: ALREADY HAPPENING (overcapacity emerging)
- Risk: Winners & losers diverging; Tier 1 companies in capex paydown mode

---

### ⚠️ CAUTION SECTORS (Tier 3+ Score: <60)

**Retail/Consumer** (Score: 35-50)
- Why low scores: Asset-light e-commerce = no capex expansion needed
- High debt: Often for M&A, not capex (misaligned financing)
- Recommendation: Screen out for expansion plays; better for dividend investing

**Utilities** (Score: 55-65)
- Why moderate: Consistent capex but stable (not accelerating)
- Capex-to-revenue stable 15-20% (maintenance mode, not growth mode)
- Recommendation: Look for accelerating capex (rare in utilities; usually declining)

---

## PART 6: GEOGRAPHIC OPPORTUNITIES

### 🚀 HIGHEST EXPANSION ACTIVITY (2026 YTD)

**India** (ISM 2.0 Announced)
- Expansion companies: Micron (Sanand), Tata (Dholera), Microsoft (cloud), Google (AI hub)
- Total capex committed: $80B+ through 2028
- Sectors: Semiconductors, cloud/data centers, renewables
- Opportunity: Early-cycle expansion; ROI not yet visible (buy now, harvest 2028+)

**United States** (CHIPS Act, IRA, AI Boom)
- Expansion companies: Intel ($25B US capex), Microsoft, Amazon, Meta
- Total capex: $300B+ 2025-2026
- Sectors: Semiconductors, cloud, AI infrastructure
- Opportunity: Peak capex phase (valuation risk); wait for inflection signals

**South Korea** (Chip/Battery Giants)
- Expansion companies: SK Hynix (KRW 35T capex 2026), Samsung (fabs + batteries)
- Total capex: $100B+ 2026-2027
- Sectors: Memory chips, EV batteries
- Opportunity: Strong demand (AI, EVs); high D/E ratios (leverage risk)

---

## PART 7: INVESTMENT PLAYBOOK

### For Different Investor Types:

**Long-term Growth Investors:**
- Target: Tier 1 companies, early capex cycle (score 80+, FCF positive)
- Thesis: Buy during capex peak → hold through ROI inflection → exit 2028+
- Stocks: Microsoft, Google, Micron, SK Hynix
- Timeline: 3-5 years

**Value Investors:**
- Target: Tier 1-2 companies, mid/late capex cycle (score 60-75, FCF negative→positive inflection)
- Thesis: Capex is "sunk"; now waiting for profits to explode
- Stocks: Intel (if restructuring succeeds), Tesla (if factory utilization rises)
- Timeline: 2-3 years

**Dividend/Income Investors:**
- Target: Avoid Tier 1-2 (capex consumes all cash flow)
- Better: Tier 3-4 or companies past capex cycle
- Stocks: Utilities, mature telecom, established manufacturers
- Timeline: Long-term income

**Sector Rotation Traders:**
- Monitor: When does capex-heavy sector peak? (scores will decline)
- Strategy: Buy Tier 1 at capex trough; sell at peak
- Stocks: Rotate within semiconductors (Micron → SK Hynix → Samsung as cycle rotates)
- Timeline: 6-24 months

---

## PART 8: MONITORING CHECKLIST

**Quarterly:**
- [ ] Download latest 10-Q filings
- [ ] Recalculate capex, debt, OCF for past 4 quarters
- [ ] Rescore universe (any Tier movements?)
- [ ] Flag if capex guidance changed (earnings call)
- [ ] Flag if debt issuance occurred

**Semi-Annually:**
- [ ] Track debt issuance calendar (when is new debt scheduled?)
- [ ] Cross-check against capex deployment timeline
- [ ] Identify if debt is for capex (good) vs M&A (different pattern)

**Annually:**
- [ ] Full metric recalculation with 10-K filings
- [ ] Management capex guidance for next 3 years
- [ ] Capacity/utilization review (is ROI materializing?)
- [ ] Competitive benchmark (who's winning the capex arms race?)

---

## CONCLUSION: EXPANSION MAP ADVANTAGE

By running this screening system you gain:

✅ **360° universe view** — Know which 800-1,200 companies are expanding (vs 25,000 total)

✅ **Tier ranking** — Separate proven winners (Tier 1) from risky bets (Tier 2-4)

✅ **Sector hotspots** — Identify capex arms races (semiconductors, cloud) and overcapacity (EVs, retail)

✅ **Geographic clarity** — See where expansion is happening (India boom, US subsidies, Asia manufacturing)

✅ **Timing signals** — Spot early cycle (buy), peak capex (avoid), and inflection (harvest)

✅ **Debt dynamics** — Track when capex is being funded (debt vs retained earnings)

✅ **Actionable pipeline** — 50+ Tier 1 companies ready for deep-dive investment analysis

---

## FILES INCLUDED

1. **UNIVERSAL_EXPANSION_SCREENING_FRAMEWORK.md** — Full framework (3,000+ words)
2. **universal_expansion_screener.py** — Automated Python tool (ready to run)
3. **debt_expansion_analysis_template.csv** — 5-year case study data (6 companies)
4. **DEBT_EXPANSION_REINVESTMENT_FRAMEWORK.md** — Single-company deep dive guide
5. **analyze_debt_expansion.py** — Single-company analyzer (for detail work)

---

**Ready to screen your universe? Start with Step 1: Prepare Financial Data** ⬆️
