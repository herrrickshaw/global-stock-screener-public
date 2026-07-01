#!/usr/bin/env python3
"""
Universal Stock Universe Expansion Screener
===========================================
Scans ~3,500 global stocks to identify expansion companies that:
1. Took on debt in last 5 years
2. Increased capex for expansion
3. Are reinvesting profits

This tool produces tier rankings, sector analysis, and investment signals.

Usage:
    python universal_expansion_screener.py [--output-dir results/]
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


class UniversalExpansionScreener:
    """Screens entire stock universe for expansion candidates"""

    # Scoring weights (totals 100)
    SCORING_WEIGHTS = {
        "debt_expansion": 20,  # D/E ratio rising
        "capex_acceleration": 20,  # Capex CAGR rising
        "profit_reinvestment": 15,  # Retained earnings growing
        "profitability_quality": 15,  # Operating income trends
        "sustainability": 15,  # FCF positive/negative
        "timing_alignment": 10,  # Debt issuance = capex timing
        "leverage_health": 5,  # Debt service coverage
    }

    # Sector-specific thresholds
    SECTOR_THRESHOLDS = {
        "Semiconductors": {"capex_revenue_min": 20, "de_rise_min": 0.1, "score_min": 75},
        "Cloud/Data Centers": {"capex_revenue_min": 12, "de_rise_min": 0.1, "score_min": 70},
        "Electric Vehicles": {"capex_revenue_min": 10, "de_rise_min": 0.05, "score_min": 60},
        "Renewable Energy": {"capex_revenue_min": 15, "de_rise_min": 0.1, "score_min": 65},
        "Telecom/5G": {"capex_revenue_min": 12, "de_rise_min": 0.05, "score_min": 60},
        "Pharmaceuticals": {"capex_revenue_min": 3, "de_rise_min": 0.0, "score_min": 50},
        "Industrials": {"capex_revenue_min": 8, "de_rise_min": 0.05, "score_min": 55},
        "Retail/Consumer": {"capex_revenue_min": 2, "de_rise_min": 0.0, "score_min": 40},
    }

    def __init__(self, data_df):
        """
        Initialize screener with financial data

        Parameters:
        -----------
        data_df : pd.DataFrame
            Columns required:
            - ticker, company_name, market_cap_b, sector, country
            - total_debt_2021-2026, total_equity_2021-2026
            - capex_2021-2026, revenue_2021-2026
            - operating_cf_2021-2026, operating_income_2021-2026
            - net_income_2021-2026, retained_earnings_2021-2026
            - dividends_paid_2021-2026
        """
        self.data = data_df.copy()
        self.scores = pd.DataFrame()
        self.results = pd.DataFrame()

    def calculate_debt_expansion_score(self, company):
        """Calculate debt expansion score (0-20 points)"""
        de_2021 = company["total_debt_2021"] / (company["total_equity_2021"] + 0.01)
        de_2026 = company["total_debt_2026"] / (company["total_equity_2026"] + 0.01)
        de_change = de_2026 - de_2021

        score = 0

        # Debt ratio rising
        if de_change > 0.10:
            score += 10
        elif de_change > 0.05:
            score += 7
        elif de_change > 0:
            score += 3

        # Absolute debt growth
        debt_ratio = company["total_debt_2026"] / (company["total_debt_2021"] + 0.01)
        if debt_ratio > 1.5:
            score += 10
        elif debt_ratio > 1.2:
            score += 7
        elif debt_ratio > 1.0:
            score += 3

        return min(score, 20)

    def calculate_capex_acceleration_score(self, company):
        """Calculate capex acceleration score (0-20 points)"""
        capex_values = [
            company["capex_2021"],
            company["capex_2022"],
            company["capex_2023"],
            company["capex_2024"],
            company["capex_2025"],
            company["capex_2026"],
        ]
        capex_values = [x for x in capex_values if x > 0]  # Remove zeros

        if len(capex_values) < 2:
            return 0

        # CAGR calculation
        capex_cagr = (
            (capex_values[-1] / capex_values[0]) ** (1 / (len(capex_values) - 1)) - 1
        ) * 100

        score = 0

        # Capex CAGR
        if capex_cagr > 25:
            score += 10
        elif capex_cagr > 15:
            score += 7
        elif capex_cagr > 5:
            score += 3

        # Capex-to-Revenue trend
        _capex_rev_2021 = (
            (company["capex_2021"] / company["revenue_2021"]) * 100
            if company["revenue_2021"] > 0
            else 0
        )
        capex_rev_2026 = (
            (company["capex_2026"] / company["revenue_2026"]) * 100
            if company["revenue_2026"] > 0
            else 0
        )

        if capex_rev_2026 > 10:
            score += 10
        elif capex_rev_2026 > 5:
            score += 7
        elif capex_rev_2026 > 2:
            score += 3

        return min(score, 20)

    def calculate_profit_reinvestment_score(self, company):
        """Calculate profit reinvestment score (0-15 points)"""
        re_2021 = company["retained_earnings_2021"]
        re_2026 = company["retained_earnings_2026"]

        # Retained earnings growth
        re_cagr = ((re_2026 / (re_2021 + 0.01)) ** (1 / 5) - 1) * 100 if re_2021 > 0 else 0

        score = 0

        if re_cagr > 10:
            score += 8
        elif re_cagr > 5:
            score += 5

        # Dividend payout ratio
        ni_avg = (company["net_income_2021"] + company["net_income_2026"]) / 2
        div_avg = (company["dividends_paid_2021"] + company["dividends_paid_2026"]) / 2
        div_pct = (div_avg / (ni_avg + 0.01)) * 100 if ni_avg > 0 else 0

        if div_pct < 30:
            score += 7
        elif div_pct < 50:
            score += 4

        return min(score, 15)

    def calculate_profitability_score(self, company):
        """Calculate profitability quality score (0-15 points)"""
        oi_2021 = company["operating_income_2021"]
        oi_2026 = company["operating_income_2026"]

        # Operating income trend
        oi_cagr = ((oi_2026 / (oi_2021 + 0.01)) ** (1 / 5) - 1) * 100 if oi_2021 > 0 else 0

        score = 0

        if oi_cagr > 10:
            score += 8
        elif oi_cagr > 5:
            score += 5

        # Operating margin stability
        om_2021 = (oi_2021 / (company["revenue_2021"] + 0.01)) * 100
        om_2026 = (oi_2026 / (company["revenue_2026"] + 0.01)) * 100
        om_change = abs(om_2026 - om_2021)

        if om_change < 2:
            score += 7
        elif om_change < 5:
            score += 3

        return min(score, 15)

    def calculate_sustainability_score(self, company):
        """Calculate sustainability score (0-15 points)"""
        fcf_2026 = company["operating_cf_2026"] - company["capex_2026"]
        fcf_2021 = company["operating_cf_2021"] - company["capex_2021"]

        score = 0

        # FCF trend
        if fcf_2026 > 0 and fcf_2026 > fcf_2021:
            score += 8
        elif fcf_2026 > 0:
            score += 5

        # Capex-to-OCF ratio
        capex_ocf_avg = (
            (company["capex_2025"] + company["capex_2026"])
            / (company["operating_cf_2025"] + company["operating_cf_2026"] + 0.01)
            * 100
        )

        if capex_ocf_avg < 70:
            score += 7
        elif capex_ocf_avg < 90:
            score += 4

        return min(score, 15)

    def calculate_timing_score(self, company):
        """Calculate capex-debt timing alignment score (0-10 points)"""
        # This is simplified; in real analysis would compare quarterly data
        de_change = (company["total_debt_2026"] - company["total_debt_2021"]) / (
            company["total_debt_2021"] + 0.01
        )
        capex_change = (company["capex_2026"] - company["capex_2021"]) / (
            company["capex_2021"] + 0.01
        )

        # If both rose significantly and proportionally
        if de_change > 0.05 and capex_change > 0.3:
            return 10
        elif de_change > 0.05 and capex_change > 0.1:
            return 7
        else:
            return 0

    def calculate_leverage_health_score(self, company):
        """Calculate leverage health score (0-5 points)"""
        # Simplified: if debt service coverage assumed 2x with current OI
        de_2026 = company["total_debt_2026"] / (company["total_equity_2026"] + 0.01)

        if de_2026 < 1.5:
            return 5
        elif de_2026 < 2.5:
            return 3
        else:
            return 0

    def calculate_total_score(self, company):
        """Calculate total expansion score (0-100)"""
        scores = {
            "debt_expansion": self.calculate_debt_expansion_score(company),
            "capex_acceleration": self.calculate_capex_acceleration_score(company),
            "profit_reinvestment": self.calculate_profit_reinvestment_score(company),
            "profitability": self.calculate_profitability_score(company),
            "sustainability": self.calculate_sustainability_score(company),
            "timing": self.calculate_timing_score(company),
            "leverage": self.calculate_leverage_health_score(company),
        }

        total = sum(scores.values())
        return total, scores

    def assign_tier(self, score, fcf_2026, sector):
        """Assign tier based on score and profitability"""
        # Sector adjustment
        _threshold = self.SECTOR_THRESHOLDS.get(sector, {}).get("score_min", 60)

        if score >= 80 and fcf_2026 > 0:
            return "TIER 1 - Proven Expander"
        elif score >= 60:
            if fcf_2026 > 0:
                return "TIER 2 - Aggressive Expander (Healthy FCF)"
            else:
                return "TIER 2 - Aggressive Expander (Risky FCF)"
        elif score >= 40:
            return "TIER 3 - Selective Expander"
        elif score >= 20:
            return "TIER 4 - Weak Signals"
        else:
            return "NO PATTERN"

    def apply_filters(self):
        """Apply disqualification filters"""
        initial_count = len(self.data)

        # Filter 1: D/E too high
        self.data = self.data[
            (self.data["total_debt_2026"] / (self.data["total_equity_2026"] + 0.01)) < 2.0
        ]

        # Filter 2: Capex too small
        self.data = self.data[
            (self.data["capex_2021"] + self.data["capex_2026"]) / 2 > 0.1
        ]  # >100M avg

        # Filter 3: Operating income too negative
        oi_neg_count = (
            (self.data["operating_income_2021"] < 0).astype(int)
            + (self.data["operating_income_2022"] < 0).astype(int)
            + (self.data["operating_income_2023"] < 0).astype(int)
        )
        self.data = self.data[oi_neg_count < 3]

        # Filter 4: FCF not completely destroyed
        self.data = self.data[
            (
                (self.data["operating_cf_2021"] - self.data["capex_2021"])
                + (self.data["operating_cf_2026"] - self.data["capex_2026"])
            )
            > -1
        ]

        # Filter 5: Liquidity (market cap > $500M)
        self.data = self.data[self.data["market_cap_b"] > 0.5]

        filtered_count = len(self.data)
        print("\nFiltering Results:")
        print(f"  Initial universe: {initial_count}")
        print(f"  After filters: {filtered_count}")
        print(
            f"  Candidates for scoring: {filtered_count} ({(filtered_count/initial_count)*100:.1f}%)"
        )

        return self.data

    def score_universe(self):
        """Score all candidates"""
        print("\nScoring expansion candidates...")

        scores_list = []
        for idx, row in self.data.iterrows():
            total_score, dimension_scores = self.calculate_total_score(row)

            fcf_2026 = row["operating_cf_2026"] - row["capex_2026"]
            tier = self.assign_tier(total_score, fcf_2026, row.get("sector", "Unknown"))

            scores_list.append(
                {
                    "ticker": row["ticker"],
                    "company": row.get("company_name", "Unknown"),
                    "sector": row.get("sector", "Unknown"),
                    "market_cap_b": row["market_cap_b"],
                    "score": total_score,
                    "tier": tier,
                    "debt_expansion": dimension_scores["debt_expansion"],
                    "capex_acceleration": dimension_scores["capex_acceleration"],
                    "profit_reinvestment": dimension_scores["profit_reinvestment"],
                    "profitability": dimension_scores["profitability"],
                    "sustainability": dimension_scores["sustainability"],
                    "timing": dimension_scores["timing"],
                    "leverage": dimension_scores["leverage"],
                    "fcf_2026": fcf_2026,
                }
            )

        self.results = pd.DataFrame(scores_list)
        print(f"✅ Scored {len(self.results)} candidates")

        return self.results

    def generate_reports(self, output_dir="expansion_results"):
        """Generate analysis reports"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nGenerating reports to {output_dir}...")

        # Report 1: Master scorecard
        scorecard = self.results.sort_values("score", ascending=False)
        scorecard.to_csv(f"{output_dir}/expansion_scorecard.csv", index=False)
        print(f"  ✅ Scorecard: {len(scorecard)} companies")

        # Report 2: Tier distribution
        tier_dist = self.results["tier"].value_counts()
        print("\nTier Distribution:")
        for tier, count in tier_dist.items():
            pct = (count / len(self.results)) * 100
            print(f"  {tier}: {count} ({pct:.1f}%)")

        # Report 3: Sector analysis
        sector_analysis = self.results.groupby("sector").agg(
            {"score": ["mean", "max", "min", "count"], "fcf_2026": "mean"}
        )
        sector_analysis.to_csv(f"{output_dir}/sector_analysis.csv")
        print(f"  ✅ Sector analysis: {len(sector_analysis)} sectors")

        # Report 4: Top 50 Tier 1 companies
        tier1 = self.results[self.results["tier"].str.contains("TIER 1")].head(50)
        tier1.to_csv(f"{output_dir}/tier1_top50.csv", index=False)
        print(f"  ✅ Tier 1 (Top 50): {len(tier1)} companies")

        # Report 5: Summary statistics
        summary = {
            "Metric": [
                "Total Candidates Scored",
                "Average Score",
                "Median Score",
                "Std Dev Score",
                "Tier 1 Count",
                "Tier 1 %",
                "Tier 2 Count",
                "Tier 2 %",
                "Tier 3+ Count",
                "Tier 3+ %",
            ],
            "Value": [
                len(self.results),
                self.results["score"].mean(),
                self.results["score"].median(),
                self.results["score"].std(),
                len(self.results[self.results["tier"].str.contains("TIER 1", na=False)]),
                f"{(len(self.results[self.results['tier'].str.contains('TIER 1', na=False)]) / len(self.results)) * 100:.1f}%",
                len(self.results[self.results["tier"].str.contains("TIER 2", na=False)]),
                f"{(len(self.results[self.results['tier'].str.contains('TIER 2', na=False)]) / len(self.results)) * 100:.1f}%",
                len(self.results[~self.results["tier"].str.contains("TIER [12]", na=False)]),
                f"{(len(self.results[~self.results['tier'].str.contains('TIER [12]', na=False)]) / len(self.results)) * 100:.1f}%",
            ],
        }
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)

        print("\n📊 Summary Statistics:")
        print(summary_df.to_string(index=False))

        print(f"\n✅ All reports saved to {output_dir}/")
        return output_path

    def run(self, output_dir="expansion_results"):
        """Run complete screening pipeline"""
        print("\n" + "=" * 80)
        print("UNIVERSAL STOCK UNIVERSE EXPANSION SCREENER")
        print(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}")
        print("=" * 80)

        # Execute pipeline
        self.apply_filters()
        results = self.score_universe()
        self.generate_reports(output_dir)

        return results


# Example usage
if __name__ == "__main__":

    # Create sample data (in real usage, this would be loaded from SEC/Yahoo Finance)
    sample_data = pd.DataFrame(
        {
            "ticker": ["MU", "INTC", "MSFT", "GOOGL", "AMZN", "META"],
            "company_name": ["Micron", "Intel", "Microsoft", "Google", "Amazon", "Meta"],
            "sector": [
                "Semiconductors",
                "Semiconductors",
                "Cloud/Data Centers",
                "Cloud/Data Centers",
                "Cloud/Data Centers",
                "Cloud/Data Centers",
            ],
            "market_cap_b": [110, 95, 550, 1200, 1600, 550],
            "total_debt_2021": [5.2, 18.5, 50.5, 10, 42.8, 15.2],
            "total_debt_2022": [6.1, 20.8, 55.2, 12, 48.5, 29.3],
            "total_debt_2023": [9.3, 35.2, 60.8, 15, 55.2, 35.8],
            "total_debt_2024": [7.8, 38.5, 65.3, 18, 62.1, 29.1],
            "total_debt_2025": [4.6, 41.2, 120.5, 25, 78.3, 59.0],
            "total_debt_2026": [4.2, 42.8, 125.8, 30, 92.5, 65.0],
            "total_equity_2021": [20.8, 62.3, 142.3, 220, 135.2, 75.8],
            "total_equity_2022": [21.2, 59.2, 148.1, 230, 142.1, 68.5],
            "total_equity_2023": [26.5, 57.8, 152.5, 240, 148.3, 62.1],
            "total_equity_2024": [30.2, 51.2, 158.2, 250, 158.5, 65.8],
            "total_equity_2025": [30.9, 48.5, 162.3, 260, 168.2, 68.2],
            "total_equity_2026": [28.2, 45.1, 165.8, 270, 175.1, 70.5],
            "capex_2021": [5.8, 14.5, 8.5, 8, 40.1, 24.1],
            "capex_2022": [6.5, 19.1, 12.1, 10, 55.3, 32.5],
            "capex_2023": [7.2, 25.8, 15.2, 12, 65.8, 36.2],
            "capex_2024": [12.4, 23.9, 22.5, 15, 85.0, 42.8],
            "capex_2025": [15.9, 14.0, 65.0, 50, 130.0, 96.0],
            "capex_2026": [25.0, 14.0, 95.0, 85, 155.0, 115.0],
            "revenue_2021": [27.4, 78.0, 168.1, 257, 469.8, 117.9],
            "revenue_2022": [30.2, 63.1, 198.3, 282, 514.2, 114.9],
            "revenue_2023": [30.8, 54.2, 198.9, 307, 575.2, 134.9],
            "revenue_2024": [32.1, 58.1, 245.1, 320, 620.4, 155.2],
            "revenue_2025": [36.5, 50.2, 380.2, 350, 665.1, 180.5],
            "revenue_2026": [39.2, 50.8, 435.5, 380, 715.2, 195.2],
            "operating_cf_2021": [14.2, 30.4, 94.3, 120, 137.5, 68.2],
            "operating_cf_2022": [11.8, 23.1, 110.1, 130, 128.5, 61.2],
            "operating_cf_2023": [12.5, 14.2, 114.2, 140, 145.2, 78.5],
            "operating_cf_2024": [18.2, 16.4, 128.5, 150, 158.3, 94.1],
            "operating_cf_2025": [22.1, 10.5, 145.8, 160, 165.2, 106.2],
            "operating_cf_2026": [28.5, 8.2, 168.2, 170, 175.5, 115.5],
            "operating_income_2021": [8.5, 24.3, 69.9, 90, 82.3, 52.8],
            "operating_income_2022": [6.2, 9.2, 82.3, 95, 78.5, 38.5],
            "operating_income_2023": [4.1, 8.4, 88.5, 100, 102.5, 58.2],
            "operating_income_2024": [12.8, 13.2, 105.0, 110, 128.5, 78.5],
            "operating_income_2025": [17.4, 8.5, 125.3, 125, 142.8, 105.3],
            "operating_income_2026": [21.5, 10.0, 140.5, 140, 158.2, 125.2],
            "net_income_2021": [6.3, 20.1, 72.8, 76, 97.2, 39.1],
            "net_income_2022": [1.5, 1.4, 84.2, 59, -2.7, -23.2],
            "net_income_2023": [-2.1, -2.2, 88.5, 59, 30.4, 23.2],
            "net_income_2024": [9.2, 9.5, 105.2, 84, 92.1, 42.5],
            "net_income_2025": [13.2, 2.3, 128.5, 97, 125.3, 78.5],
            "net_income_2026": [17.5, 3.5, 142.8, 113, 142.5, 95.8],
            "retained_earnings_2021": [34.7, 95.2, 215.3, 280, 185.3, 95.3],
            "retained_earnings_2022": [36.8, 88.3, 226.5, 300, 187.2, 78.9],
            "retained_earnings_2023": [52.1, 89.4, 248.3, 350, 192.8, 85.4],
            "retained_earnings_2024": [73.5, 92.1, 268.9, 400, 208.5, 105.8],
            "retained_earnings_2025": [78.4, 94.8, 315.2, 480, 228.5, 128.5],
            "retained_earnings_2026": [94.7, 97.2, 365.5, 560, 252.8, 155.3],
            "dividends_paid_2021": [0.4, 6.1, 21.7, 0, 0.0, 0.0],
            "dividends_paid_2022": [0.4, 5.6, 24.0, 0, 0.0, 0.0],
            "dividends_paid_2023": [0.4, 5.6, 25.5, 0, 0.0, 0.0],
            "dividends_paid_2024": [0.4, 6.0, 30.0, 0, 0.0, 0.0],
            "dividends_paid_2025": [0.4, 4.2, 38.5, 0, 0.0, 0.0],
            "dividends_paid_2026": [0.4, 2.0, 42.0, 0, 0.0, 0.0],
        }
    )

    screener = UniversalExpansionScreener(sample_data)
    results = screener.run(
        output_dir="/private/tmp/claude-501/-Users-umashankar/7c1286ad-5a8d-41f6-8f3d-6f5a2866ae33/scratchpad/expansion_results"
    )

    print("\n✅ Screening complete!")
    print("\nTop 10 Expansion Companies by Score:")
    print(
        results.nlargest(10, "score")[["ticker", "company", "sector", "score", "tier"]].to_string(
            index=False
        )
    )
