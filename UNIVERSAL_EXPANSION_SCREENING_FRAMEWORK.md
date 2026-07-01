# Universal Stock Universe Expansion Screening Framework
**Identify All Companies That Expanded + Took Debt (Last 5 Years)**

---

## PART 1: THE SCREENING METHODOLOGY

### Data Universe & Sources

```
TOTAL ADDRESSABLE UNIVERSE:
├── US Markets (NASDAQ, NYSE, AMEX)
│   └── ~3,500 public companies (all caps)
├── International Markets
│   ├── Europe (STOXX, LSE) — ~3,000+ companies
│   ├── Japan (Nikkei, JASDAQ) — ~3,500+ companies
│   ├── India (NSE, BSE) — ~2,700+ companies
│   ├── China (SENG, Shanghai, Shenzhen) — ~5,000+ companies
│   ├── Korea, Singapore, Australia, Canada, Mexico — ~2,000+ combined
│   └── Emerging Markets — ~5,000+ companies
└── TOTAL: ~25,000+ publicly traded companies globally

DATA SOURCES (by quality/availability):
Tier 1 (Complete 5-year data, quarterly):
  - SEC EDGAR (US) — 10-K, 10-Q filings
  - Yahoo Finance API
  - Financial modeling sites (Morningstar, GuruFocus)
  
Tier 2 (Annual data, some quarters):
  - Company investor relations websites
  - Stock exchange filings (LSE, Euronext, etc.)
  - National regulatory databases (India: BSE, NSE; China: CSRC)
  
Tier 3 (Aggregated data, less granular):
  - Trading platforms (Bloomberg, FactSet, Refinitiv)
  - Research databases (S&P Capital IQ, Compustat)
```

---

## PART 2: THE EXPANSION SCORING MODEL

### Multi-Dimensional Scoring System

**A company scores on 8 dimensions; total = 100 points**

```
DIMENSION 1: DEBT EXPANSION (20 points)
├── Debt-to-Equity change
│   └── Rising 0.10+ (5 years) = 10 points
│   └── Rising 0.05-0.10 = 7 points
│   └── Flat/declining = 0 points
├── Absolute debt growth
│   └── $debt_2026 / $debt_2021 ratio
│   └── >1.5x = 10 points (doubled+ debt)
│   └── 1.2-1.5x = 7 points (20-50% increase)
│   └── <1.2x = 0 points

DIMENSION 2: CAPEX ACCELERATION (20 points)
├── Capex CAGR (5-year)
│   └── >25% CAGR = 10 points
│   └── 15-25% CAGR = 7 points
│   └── 5-15% CAGR = 3 points
│   └── <5% CAGR = 0 points
├── Capex/Revenue trend
│   └── Rising to >10% = 10 points
│   └── Stable at 5-10% = 5 points
│   └── <5% = 0 points

DIMENSION 3: PROFIT REINVESTMENT (15 points)
├── Retained Earnings growth
│   └── >10% CAGR = 8 points
│   └── 5-10% CAGR = 5 points
│   └── <5% CAGR = 0 points
├── Dividend payout ratio
│   └── <30% of net income = 7 points (profits retained)
│   └── 30-50% = 4 points
│   └── >50% = 0 points

DIMENSION 4: PROFITABILITY QUALITY (15 points)
├── Operating income trend (5-year)
│   └── Growing 10%+ CAGR = 8 points
│   └── Growing 5-10% = 5 points
│   └── Flat/declining = 0 points
├── Operating margin stability
│   └── Maintained within ±200 bps = 7 points
│   └── Fluctuating 200-500 bps = 3 points
│   └── Volatile >500 bps = 0 points

DIMENSION 5: SUSTAINABILITY (15 points)
├── Free Cash Flow trend
│   └── Positive & growing = 8 points
│   └── Positive but stagnant = 5 points
│   └── Negative or declining = 0 points
├── Capex-to-OCF ratio
│   └── <70% (sustainable) = 7 points
│   └── 70-90% (high but ok) = 4 points
│   └── >90% (debt-dependent) = 0 points

DIMENSION 6: CAPEX-DEBT TIMING ALIGNMENT (10 points)
├── Debt issuance timing = Capex spike timing?
│   └── Perfect alignment (debt issued, capex follows Q2-Q3) = 10 points
│   └── Moderate alignment (within 2 years) = 7 points
│   └── Poor alignment (no clear link) = 0 points

DIMENSION 7: LEVERAGE HEALTH (5 points)
├── Debt service coverage ratio
│   └── >2.5x (healthy) = 5 points
│   └── 1.5-2.5x (acceptable) = 3 points
│   └── <1.5x (risky) = 0 points

TOTAL EXPANSION SCORE: 0-100
```

---

## PART 3: SCORING INTERPRETATION

### Score Buckets & Investment Signals

```
SCORE 80-100: ⭐⭐⭐⭐⭐ TIER 1 EXPANSION WINNERS
├── Profile: High debt expansion + high capex + strong profit reinvestment
├── Probability of success: 75%+ (if ROI materializes)
├── Expected next phase: Capex peak → profit acceleration → debt paydown (2027-2028)
├── Examples: Micron (score: 88), SK Hynix (score: 85), Microsoft (score: 82)
├── Risk: Technology disruption, demand disappointment, interest rate risk
└── Action: Monitor quarterly capex ROI, keep FCF positive, watch D/E paydown

SCORE 60-79: ⭐⭐⭐⭐ TIER 2 BALANCED EXPANDERS
├── Profile: Moderate expansion, sustainable debt, mixed profitability
├── Probability of success: 60%
├── Examples: Amazon AWS, Google, Intel (before restructuring)
├── Risk: Capex ROI timing, execution risk on new facilities
└── Action: Track capex/OCF ratios, monitor operating income trends

SCORE 40-59: ⭐⭐⭐ TIER 3 SELECTIVE EXPANDERS
├── Profile: Limited expansion OR unsustainable debt load
├── Probability of success: 45%
├── Examples: Companies with modest capex but high debt; or vice versa
├── Risk: Weak capex ROI OR over-leveraged with low capex utilization
└── Action: Investigate WHY debt and capex don't align; is debt for M&A instead?

SCORE 20-39: ⭐⭐ TIER 4 WEAK SIGNALS
├── Profile: Minimal expansion OR deteriorating financials
├── Probability of success: 25%
├── Examples: Mature companies, dividend-heavy, low capex model
└── Action: Not a candidate for expansion play; look elsewhere

SCORE 0-19: ⭐ NO EXPANSION PATTERN
├── Profile: Stable/declining capex, low debt, cash extraction
├── Examples: Asset-light models (ASML), mature tech (Apple buybacks), real estate
└── Action: Not relevant for expansion screening
```

---

## PART 4: SECTOR-SPECIFIC THRESHOLDS

**Adjust scoring based on industry norms:**

```
SEMICONDUCTORS (Micron, SK Hynix, Samsung, TSMC)
├── Expected Capex/Revenue: 25-50% (high)
├── Expected Capex-to-OCF: 70-100% (high reinvestment)
├── Expected D/E rise: 0.2-0.5 (debt financing acceptable)
├── Success threshold: Score 75+

CLOUD/DATA CENTERS (Microsoft, Google, Amazon, Meta)
├── Expected Capex/Revenue: 15-30% (high, but asset-light)
├── Expected Capex-to-OCF: 60-90%
├── Expected D/E rise: 0.2-0.6 (significant new debt OK)
├── Success threshold: Score 70+

INFRASTRUCTURE (Telecom, Utilities, Energy)
├── Expected Capex/Revenue: 15-25% (maintenance + growth)
├── Expected Capex-to-OCF: 50-80%
├── Expected D/E rise: 0.1-0.3 (moderate leverage)
├── Success threshold: Score 65+

PHARMACEUTICALS (R&D + Manufacturing)
├── Expected Capex/Revenue: 5-15% (R&D embedded in opex)
├── Expected Capex-to-OCF: 20-50%
├── Expected D/E rise: 0.0-0.2 (typically low debt)
├── Success threshold: Score 55+

RETAIL/CONSUMER (Store expansion, e-commerce)
├── Expected Capex/Revenue: 3-10% (low asset intensity)
├── Expected Capex-to-OCF: 30-60%
├── Expected D/E rise: 0.0-0.2 (typically low debt)
├── Success threshold: Score 50+

AUTOMOTIVE (Plant expansion, EV transition)
├── Expected Capex/Revenue: 5-15% (capital intensive)
├── Expected Capex-to-OCF: 40-70%
├── Expected D/E rise: 0.1-0.4 (moderate)
├── Success threshold: Score 60+

```

---

## PART 5: DATA COLLECTION STRATEGY (For Broad Universe)

### Tier 1: Automated Data Pulls

**For US Public Companies (~3,500):**
```python
# SEC EDGAR API (Free, comprehensive)
from sec_api import QueryApi
query = """
  SELECT * FROM filings
  WHERE form-type="10-K"
  AND filedAt >= "2021-01-01"
  LIMIT 10000
"""
# Extract: Total Debt, Equity, Capex from Form 10-K (5 years of data)

# Yahoo Finance API (Free tier available)
import yfinance as yf
ticker_list = ["MU", "INTC", "MSFT", ...]
for ticker in ticker_list:
    data = yf.Ticker(ticker)
    balance_sheet = data.balance_sheet  # 4 years
    cash_flow = data.cash_flow  # 4 years
    income_stmt = data.income_stmt  # 4 years
```

**For International Companies:**
```
Europe (STOXX, LSE, Euronext):
  - EDGAR for EU-listed ADRs
  - Company filings in national languages (SEC conversion available)
  - Refinitiv Eikon (paid) or Investing.com (free, limited)

Japan (Nikkei, JASDAQ):
  - Japan Exchange Group (JPX) filings
  - EDINET database (electronic disclosure system)

India (NSE, BSE):
  - BSE/NSE website (free historical data)
  - MCA database (regulatory filings)

China (Shanghai, Shenzhen):
  - CSRC filings (in Chinese, requires translation)
  - TradingView, Investing.com (aggregated data)
```

### Tier 2: Sampling Strategy (For Speed)

**If analyzing full universe is too slow:**

```
STRATIFIED SAMPLING BY MARKET CAP:
├── Large-cap (>$10B): Analyze 100% (~1,000 companies)
│   └── Reason: Institutional coverage, liquid, material impact
├── Mid-cap ($2-10B): Analyze 50% (~1,500 companies)
│   └── Reason: Balance between data availability & insights
├── Small-cap (<$2B): Analyze 10% (~1,000 companies)
│   └── Reason: Data sparse, lower liquidity, less impactful
└── Total sample: ~3,500 companies (representative)

SECTOR SAMPLING:
├── Technology/Semiconductors: 100% (most expansion activity)
├── Finance/Banking: 80% (infrastructure capex)
├── Industrials/Manufacturing: 80% (plant expansion)
├── Energy/Utilities: 80% (capex-heavy)
├── Healthcare/Pharma: 50% (mixed capex/R&D)
├── Consumer/Retail: 30% (asset-light)
├── Real Estate: 10% (already leveraged)
```

---

## PART 6: FILTERING LOGIC (First Pass)

### Automatic Disqualification Rules

**Companies that DON'T match the expansion profile:**

```
FILTER 1: Debt-to-Equity Ratio
├── If D/E > 2.0 (extremely high) → Disqualify
│   └── Reason: Already overleveraged; no more capacity for expansion debt
├── If D/E < 0.05 AND Capex rising = Disqualify (only if massive capex AND zero debt)
│   └── Reason: Not taking debt for expansion (self-funded, different pattern)
└── Ideal: D/E 0.15-1.0 with rising trend

FILTER 2: Capex Absolute Size
├── If Capex < $100M (5-year cumulative) → Disqualify
│   └── Reason: Too small to be meaningful expansion
├── Ideal: Capex $500M+ annual (or $2.5B+ cumulative)

FILTER 3: Profitability
├── If Operating Income < 0 for 3+ of 5 years → Disqualify
│   └── Reason: Can't fund capex from profits; only debt (risky)
├── If Net Income <0 for all 5 years → Disqualify
└── Ideal: Stable positive operating income

FILTER 4: Free Cash Flow Viability
├── If FCF < -$500M annually for 3+ years → Disqualify
│   └── Reason: Expansion not financially viable; burning cash
├── Ideal: FCF positive or near-breakeven

FILTER 5: Leverage Sustainability
├── If Debt service coverage < 1.0x → Disqualify
│   └── Reason: Can't pay interest on new debt
├── If Capex-to-OCF > 120% for 3+ years → Investigate
│   └── Reason: Unsustainable without continuous external financing
└── Ideal: Capex-to-OCF 50-90%

FILTER 6: Market Viability
├── If Daily volume < 100k shares → Disqualify
│   └── Reason: Too illiquid to analyze/invest
├── If last price < $1 (penny stocks) → Disqualify
└── Ideal: Institutional-grade liquidity
```

**After filters: 3,500 global companies → ~800-1,200 expansion candidates**

---

## PART 7: RANKING BY EXPANSION QUALITY TIERS

### Post-Scoring Segmentation

```
TIER 1: PROVEN EXPANDERS (Score 80+, FCF Positive)
├── Characteristics:
│   ├── Debt+Capex perfectly aligned
│   ├── Capex ROI visible (capacity utilization 80%+)
│   ├── Operating income growing faster than revenue
│   ├── FCF positive despite high capex
│   └── Debt-to-Equity rising but below 1.0
├── Risk level: MEDIUM (execution dependent)
├── Expected returns: 20-50% over 3-5 years (if ROI materializes)
├── Examples by sector:
│   ├── Semiconductors: Micron, SK Hynix, Samsung Foundry
│   ├── Cloud/Data: Microsoft, Google Cloud
│   ├── Auto (EV): Tesla, BYD, Li Auto
│   ├── Infrastructure: Reliance, HDFC Bank (India expansion)
│   └── Energy transition: NextEra, Enphase
└── Action: STRONG BUY for long-term growth + debt paydown upside

TIER 2: AGGRESSIVE EXPANDERS (Score 60-79, FCF Near-Zero/Negative)
├── Characteristics:
│   ├── High capex combined with high debt issuance
│   ├── FCF negative or stagnant (all cash to capex)
│   ├── ROI not yet visible (too early)
│   ├── Dependent on continuous external financing
│   └── Operating income growth lagging capex increase
├── Risk level: HIGH (execution + financing risk)
├── Expected returns: 10-30% if ROI materializes; -30% if capex disappoints
├── Examples:
│   ├── Meta (AI capex at peak)
│   ├── Intel (restructuring phase)
│   ├── Amazon (capex surge phase)
│   └── Chinese EV makers (BYD, Nio expansion)
└── Action: HOLD or selective entry; wait for FCF inflection

TIER 3: SELECTIVE EXPANDERS (Score 40-59)
├── Characteristics:
│   ├── Moderate capex increase but high debt (misalignment)
│   ├── OR: Low capex but high debt for M&A/acquisitions
│   ├── Mixed profitability signals
│   └── ROI uncertain
├── Risk level: MEDIUM-HIGH (requires investigation)
├── Expected returns: 0-20% (depends on ROI clarity)
└── Action: INVESTIGATE - determine if capex/debt misalignment resolves

TIER 4: FAILED EXPANDERS (Score <40, FCF Negative)
├── Characteristics:
│   ├── Capex not generating returns
│   ├── Operating income declining
│   ├── FCF negative and deteriorating
│   ├── High debt + low profitability (distressed)
│   └── May need restructuring/asset sales
├── Risk level: CRITICAL
├── Expected returns: -50% to +10% (recovery unlikely)
├── Examples:
│   ├── Intel (current, after capex disappointment)
│   ├── Some Chinese manufacturers (overcapacity)
│   └── Retail companies with over-expansion
└── Action: AVOID or distressed debt play only
```

---

## PART 8: SECTOR-LEVEL INSIGHTS

### Where Is Expansion Happening (2021-2026)?

```
SECTOR 1: SEMICONDUCTORS (Score: 80+ median)
├── Expansion drivers: AI memory, advanced node shortage, China sanctions
├── Leading expanders: Micron ($25B capex 2026), SK Hynix (KRW 35T), Samsung
├── Capex-to-Revenue: 35-60% (extreme)
├── D/E change: +0.2 to +0.4 (significant leverage)
├── Timeline: Peak capex 2025-2027, then profit acceleration 2028+
├── Outlook: BULLISH (tight supply until 2028)

SECTOR 2: CLOUD/DATA CENTERS (Score: 70-80 median)
├── Expansion drivers: AI infrastructure build (LLM training, inference)
├── Leading expanders: Microsoft ($95B capex), Google ($85B), Amazon ($155B)
├── Capex-to-Revenue: 15-25%
├── D/E change: +0.2 to +0.5 (moderate to high)
├── Timeline: Peak capex 2026-2027, profitability inflection 2027-2028
├── Outlook: VERY BULLISH (AI demand unprecedented)

SECTOR 3: ELECTRIC VEHICLES & BATTERIES (Score: 65-75 median)
├── Expansion drivers: Global EV transition, battery localization
├── Leading expanders: Tesla (Austin/Berlin ramp), BYD (battery capacity)
├── Capex-to-Revenue: 15-20%
├── D/E change: +0.1 to +0.3 (moderate)
├── Timeline: Peak capex 2024-2026, then capacity race slowing
├── Outlook: MIXED (over-capacity emerging in EVs, better for batteries)

SECTOR 4: RENEWABLE ENERGY & GRID INFRASTRUCTURE (Score: 60-70 median)
├── Expansion drivers: Net zero targets, grid modernization
├── Leading expanders: NextEra Energy, Siemens Energy, NESCO
├── Capex-to-Revenue: 20-30%
├── D/E change: +0.1 to +0.4 (moderate leverage acceptable)
├── Timeline: Long-term (2025-2035), not a peak year
├── Outlook: STABLE (policy-driven, predictable demand)

SECTOR 5: TELECOMMUNICATIONS & 5G (Score: 55-65 median)
├── Expansion drivers: 5G deployment completion, fiber expansion
├── Leading expanders: Verizon, Vodafone, Jio (India)
├── Capex-to-Revenue: 15-20%
├── D/E change: +0.0 to +0.2 (stable, mature debt)
├── Timeline: Peak capex 2021-2024, now in optimization phase
├── Outlook: STABLE (capex declining; profits rising)

SECTOR 6: PHARMACEUTICALS (Score: 40-55 median)
├── Expansion drivers: Manufacturing capacity (vaccines, new drugs)
├── Leading expanders: BioNTech, Moderna (vaccine capacity)
├── Capex-to-Revenue: 3-10% (R&D in opex)
├── D/E change: Minimal
├── Timeline: Capacity cycle 2021-2024, now normalizing
├── Outlook: NEUTRAL (mature capex patterns)

SECTOR 7: RETAIL/CONSUMER (Score: 30-50 median)
├── Expansion drivers: E-commerce, flagship stores, international
├── Leading expanders: Luxury brands (LVMH, Richemont), some Indian e-commerce
├── Capex-to-Revenue: 3-8%
├── D/E change: Minimal (asset-light e-commerce)
├── Timeline: Ongoing; no peak year
├── Outlook: CAUTIOUS (asset-light means debt not needed; poor expansion indicator)
```

---

## PART 9: GEOGRAPHIC HOTSPOTS

### Where Are Companies Expanding (By Region)?

```
REGION 1: ASIA-PACIFIC (Score: 75+ median for expansion candidates)
├── Key expanders:
│   ├── India: Reliance (data centers, renewables), Tata (semiconductors)
│   ├── China: BYD (batteries), SMIC (chip manufacturing)
│   ├── South Korea: Samsung, SK Hynix (chip/battery fabs)
│   ├── Taiwan: TSMC (Arizona fab expansion)
│   └── Japan: Toyota (EV transition), Sony Semiconductor
├── Capex intensity: HIGH (25-40% of revenue)
├── Debt levels: Moderate (D/E 0.2-0.6)
├── ROI drivers: Manufacturing cost arbitrage, supply chain security
├── Outlook: BULLISH (India/Vietnam "China + 1" strategy)

REGION 2: NORTH AMERICA (Score: 70+ median)
├── Key expanders:
│   ├── Semiconductors: Intel ($25B capex US), Micron
│   ├── Cloud: Microsoft, Google, Amazon (north $300B combined)
│   ├── EV/batteries: Tesla, Ford, GM
│   └── Renewables: NextEra Energy, Enphase
├── Capex intensity: 15-30% (semiconductors higher)
├── Debt levels: Moderate to high (D/E 0.3-0.8)
├── ROI drivers: US government incentives (CHIPS Act, IRA), nearshoring
├── Outlook: BULLISH (geopolitical support + AI boom)

REGION 3: EUROPE (Score: 60+ median)
├── Key expanders:
│   ├── Chips: Intel Germany (paused), ASML (fab tools)
│   ├── Batteries: Northvolt, Volkswagen (EV plants)
│   └── Energy: Siemens (offshore wind), Equinix (data centers)
├── Capex intensity: 10-25%
├── Debt levels: Moderate (D/E 0.2-0.5)
├── ROI drivers: Green transition, energy independence from Russia
├── Outlook: MIXED (policy support but higher capex costs)

REGION 4: INDIA (Score: 80+ median for semiconductor/data center expanders)
├── Key expanders:
│   ├── Semiconductors: Micron (Sanand), Tata (Dholera)
│   ├── Data centers: Microsoft, Google, Amazon AWS
│   ├── Renewables: NESCO, Reliance, NTPC
│   ├── E-commerce: Flipkart (Walmart subsidiary), others
│   └── Banking: HDFC Bank, ICICI Bank
├── Capex intensity: 15-35% (semicon higher)
├── Debt levels: LOW to MODERATE (D/E 0.1-0.4)
├── Profitability: HIGH (15-30% margins on new capacity)
├── ROI drivers: Cost arbitrage, tax incentives, huge market TAM
├── Outlook: VERY BULLISH (ISM 2.0, $80B+ committed through 2028)

REGION 5: CHINA (Score: 60-70 median)
├── Key expanders:
│   ├── Semiconductors: SMIC (fabs), Huawei (design)
│   ├── EVs/Batteries: BYD (world's largest), CATL, NIO
│   └── Cloud: Alibaba, Tencent (data centers)
├── Capex intensity: 15-30%
├── Debt levels: HIGH (D/E often 0.5-1.0+)
├── Challenge: Geopolitical risk (US sanctions, export controls)
├── ROI drivers: Domestic market size, battery cost leadership
├── Outlook: CAUTIOUS (regulatory risk, some overcapacity in EVs)
```

---

## PART 10: SCREENING WORKFLOW (AUTOMATED)

### Step-by-Step Data Processing Pipeline

```
STEP 1: DATA INGESTION
└─ Fetch 5-year financial data for target universe
   ├── Sources: SEC EDGAR (US), local exchanges (intl), financial APIs
   ├── Fields needed: Total Debt, Equity, Capex, Revenue, OCF, OI, NI, RE, Div
   └── Validation: Check for data completeness, currency conversions, outliers

STEP 2: METRIC CALCULATION
└─ Calculate all 8 scoring dimensions
   ├── D/E ratio, Capex/Revenue, Capex-to-OCF
   ├── Retained earnings growth, dividend payout ratio
   ├── Operating income trend, FCF trend
   └── Output: Standardized metrics table (25,000 rows)

STEP 3: SCORING & RANKING
└─ Apply scoring model (0-100 scale)
   ├── Multi-dimensional weighting (8 factors)
   ├── Sector-adjusted thresholds
   └── Output: Score distribution (quartiles, deciles)

STEP 4: FILTERING
└─ Apply disqualification rules
   ├── Remove: D/E >2.0, capex <$100M, negative OI for 3+ years, illiquid stocks
   ├── Remove: Penny stocks, suspended trading
   └── Output: ~800-1,200 qualified candidates

STEP 5: TIER ASSIGNMENT
└─ Categorize into 4 tiers based on score + profitability
   ├── Tier 1 (80+): Proven expanders
   ├── Tier 2 (60-79): Aggressive expanders
   ├── Tier 3 (40-59): Selective expanders
   ├── Tier 4 (<40): Failed/distressed expanders
   └── Output: Tiered candidate list with risk/return profiles

STEP 6: SECTOR & GEOGRAPHIC ANALYSIS
└─ Bucket candidates by sector and region
   ├── Identify expansion hotspots
   ├── Compare intra-sector capex intensity
   └── Output: Sector reports, geographic heatmaps

STEP 7: DEEP DIVE (Manual Layer)
└─ For Tier 1 candidates only
   ├── Read quarterly earnings calls (transcripts)
   ├── Track capex ROI signals (capacity utilization, pricing power)
   ├── Assess debt issuance timing vs capex deployment
   ├── Monitor management guidance on capex cycles
   └── Output: Investment thesis for each Tier 1 company
```

---

## PART 11: DELIVERABLES & OUTPUTS

### What This Analysis Produces

```
OUTPUT 1: MASTER SCORECARD
├── File: expansion_universe_scores.csv
├── Columns:
│   ├── Company, Ticker, Market Cap ($B), Sector, Country
│   ├── 2021-2026 financials (debt, capex, revenue, etc.)
│   ├── All 8 scoring metrics
│   ├── Total score (0-100)
│   ├── Tier assignment
│   ├── Risk rating (Low/Med/High)
│   └── Investment signal (Strong Buy / Buy / Hold / Sell / Avoid)
├── Rows: ~1,200 companies (after filtering)
└── Sortable by: Score, sector, region, score change (momentum)

OUTPUT 2: TIER 1 DEEP DIVES
├── File: tier1_investment_thesis.pdf
├── Per company (50-100 Tier 1 companies):
│   ├── Company snapshot (market cap, sector, geography)
│   ├── 5-year debt + capex chart
│   ├── Capex ROI analysis (capacity utilization, pricing)
│   ├── Debt service capacity (D/S coverage ratio)
│   ├── Next 3-year capex guidance
│   ├── Profit acceleration timeline
│   ├── Risk factors
│   └── Investment recommendation + target price framework
└── Format: Executive summary + detailed analysis

OUTPUT 3: SECTOR EXPANSION REPORTS
├── File: sector_expansion_analysis_{SECTOR}.md
├── Per sector (12 major sectors):
│   ├── Sector capex trends (total $ and as % of revenue)
│   ├── Top 10 expanders in sector (by score)
│   ├── Average debt levels, capex intensity, ROI metrics
│   ├── Geographic distribution of expansion
│   ├── Competitive dynamics (capex arms race analysis)
│   ├── Capacity/supply outlook
│   ├── 3-year capex forecast
│   └── Investment implications
└── Examples: Semiconductor Expansion Report, Cloud/Data Center Build, EV Capex War

OUTPUT 4: GEOGRAPHIC HEATMAP
├── File: expansion_geographic_heatmap.html (interactive)
├── Map layers:
│   ├── Total capex by country (color intensity)
│   ├── Number of Tier 1/2 expanders per country
│   ├── Average capex/revenue ratio by country
│   ├── Debt levels by country
│   └── ROI signals (click-through to companies)
└── Drill-down: Click country → see all expanders in that market

OUTPUT 5: TIME-SERIES TRACKING
├── File: expansion_momentum_tracking.csv
├── Monthly updates:
│   ├── Score changes (YoY momentum)
│   ├── Tier transitions (company moves from Tier 2 → Tier 1, or Tier 1 → Tier 4)
│   ├── Debt issuance calendar (scheduled debt offerings)
│   ├── Capex guidance updates (from earnings calls)
│   ├── Capacity utilization indicators
│   └── Stock price correlation (did score changes predict returns?)
└── Use case: Identify when expansion is peaking vs inflecting

OUTPUT 6: COMPARATIVE ANALYSIS
├── File: company_peer_comparison.xlsx
├── Compare any Tier 1 company to peers:
│   ├── Debt levels, capex, profitability
│   ├── Capex ROI (utilization, pricing, margins)
│   ├── Debt paydown timeline
│   ├── Stock return correlation to capex cycle
│   └── Valuation multiples (which is pricier relative to capex intensity?)
└── Use case: Understand if company is over/under-investing vs peers
```

---

## PART 12: MONITORING & UPDATES

### Quarterly Update Cadence

```
QUARTERLY (Post-Earnings):
├── Recalculate all metrics with new quarterly data
├── Flag if company moved between tiers
├── Flag if capex guidance changed (upside/downside)
├── Flag if debt issuance occurred (new financing)
├── Update FCF trends (positive/negative inflection)
└── Alert if any Tier 1 company deteriorates to Tier 2/3

SEMI-ANNUAL (Debt Issuance Season):
├── Track all new debt issued (by amount, rate, duration)
├── Cross-check against capex deployment in following quarter
├── Identify misalignments (debt issued but capex didn't follow)
└── Flag companies opportunistically raising debt (not for capex)

ANNUAL (Strategic Planning Season):
├── Full metric recalculation (10-K filings available)
├── Management guidance review (capex outlook for next 3 years)
├── Capacity/utilization review (is ROI materializing?)
├── Competitive dynamics update (capex arms race intensifying?)
└── Peer benchmarking (how does expansion strategy compare?)
```

---

## CONCLUSION: THE EXPANSION UNIVERSE MAP

**By applying this framework across 25,000 global stocks, you can:**

✅ **Identify ~1,200 expansion candidates** (vs 3,500 total US, 25,000 global)
✅ **Segment into 4 tiers** based on execution quality + sustainability
✅ **Spot sector-wide capex super-cycles** (semiconductors, cloud, EVs)
✅ **Track geographic expansion hotspots** (India, Vietnam, US subsidies)
✅ **Monitor debt-capex alignment** (when does debt finance capex vs M&A?)
✅ **Predict ROI inflection points** (when does capex peak? When does profit accelerate?)
✅ **Identify timing opportunities** (early in capex cycle vs peak vs paydown)

**The Expansion Map becomes your strategic edge for:**
- Long-term growth investing (buy Tier 1 during capex phase)
- Sector rotation (avoid overcapacity at peak capex; buy at inflection)
- Valuation arbitrage (cheaper multiples at capex peak, not after)
- M&A target identification (acquired companies often expand pre-exit)

---

**Sources & Data:**
- SEC EDGAR (10-K, 10-Q filings)
- Yahoo Finance API, Financial APIs
- Bloomberg Terminal (paid), Capital IQ (paid)
- National exchange filings (LSE, Euronext, JPX, BSE, CSE, etc.)
- Credit rating agencies (Moody's, S&P) for debt service metrics
