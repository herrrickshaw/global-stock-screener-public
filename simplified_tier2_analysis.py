#!/usr/bin/env python3
"""
Simplified Tier 2 Analysis Framework
====================================
Fast deep-dive metrics for analyzing aggressive expanders.
Reduces from full 8 dimensions to 4 actionable investment signals.

Focus: Capex ROI, Debt Paydown Timeline, Margin Expansion Potential,
        Competitive Positioning
"""

from typing import Dict, Tuple

import pandas as pd


class SimplifiedTier2Analyzer:
    """Analyzes Tier 2 candidates with 4 key investment signals"""

    def __init__(self, tier2_df):
        """Initialize with Tier 2 screener results"""
        self.tier2 = tier2_df.copy()
        self.results = []

    def analyze_capex_roi(self, company: Dict) -> Tuple[float, str]:
        """
        Signal 1: Capex ROI & Capacity Utilization

        Metrics:
        - Operating income growth vs capex growth (ratio)
        - Revenue growth vs capex spend (efficiency)
        - OI margin trend (profit per capex dollar)

        Scoring:
        - >2.0x: OI growing faster than capex (ROI inflection) → 100pts
        - 1.5-2.0x: Healthy ROI profile                      → 75pts
        - 1.0-1.5x: Capex catching up to revenue             → 50pts
        - <1.0x: ROI still negative (peak capex phase)        → 25pts
        """

        # OI growth vs capex growth
        oi_2021 = company.get("operating_income_2021", 0)
        oi_2026 = company.get("operating_income_2026", 0)
        capex_2021 = company.get("capex_2021", 0.1)
        capex_2026 = company.get("capex_2026", 0.1)

        if oi_2021 > 0 and capex_2021 > 0:
            oi_growth = ((oi_2026 / oi_2021) ** (1 / 5) - 1) * 100
            capex_growth = ((capex_2026 / capex_2021) ** (1 / 5) - 1) * 100
            roi_ratio = (oi_growth + 0.1) / (capex_growth + 0.1)
        else:
            roi_ratio = 0.5

        # Revenue per capex dollar
        revenue_2026 = company.get("revenue_2026", 1)
        _capex_intensity = capex_2026 / (revenue_2026 + 0.01) * 100

        # OI margin expansion
        oi_margin_2021 = oi_2021 / (company.get("revenue_2021", 1) + 0.01) * 100
        oi_margin_2026 = oi_2026 / (revenue_2026 + 0.01) * 100
        _margin_delta = oi_margin_2026 - oi_margin_2021

        if roi_ratio > 2.0:
            score = 100
            phase = "ROI Inflection (profits accelerating)"
        elif roi_ratio > 1.5:
            score = 75
            phase = "Healthy ROI (balanced growth)"
        elif roi_ratio > 1.0:
            score = 50
            phase = "Capex Phase (investment heavy)"
        else:
            score = 25
            phase = "Peak Capex (low near-term ROI)"

        return score, phase

    def analyze_debt_paydown(self, company: Dict) -> Tuple[float, str, float]:
        """
        Signal 2: Debt Paydown Timeline & FCF Generation

        Metrics:
        - Net debt reduction capability (FCF after capex)
        - Debt-to-FCF ratio (years to paydown)
        - Interest coverage (OI/Interest)

        Timeline Classification:
        - <2 years to paydown: Aggressive payoff  → 100pts
        - 2-4 years: Moderate timeline            → 75pts
        - 4-7 years: Extended expansion phase     → 50pts
        - >7 years: Debt burden                   → 25pts
        """

        debt_2026 = company.get("total_debt_2026", 0.1)
        fcf_2021 = company.get("operating_cf_2021", 0.1) - company.get("capex_2021", 0.01)
        fcf_2026 = company.get("operating_cf_2026", 0.1) - company.get("capex_2026", 0.01)
        fcf_avg = (fcf_2021 + fcf_2026) / 2

        if fcf_avg > 0:
            paydown_years = debt_2026 / (fcf_avg + 0.01)
        else:
            paydown_years = 999

        if paydown_years <= 2:
            score = 100
            timeline = "Aggressive Payoff (<2 years)"
        elif paydown_years <= 4:
            score = 75
            timeline = "Moderate Timeline (2-4 years)"
        elif paydown_years <= 7:
            score = 50
            timeline = "Extended Phase (4-7 years)"
        else:
            score = 25
            timeline = "Long-term Debt Cycle (>7 years)"

        return score, timeline, min(paydown_years, 10)

    def analyze_margin_expansion(self, company: Dict) -> Tuple[float, str]:
        """
        Signal 3: Margin Expansion Potential

        Metrics:
        - Operating margin trend (OI/revenue)
        - Net margin trend (NI/revenue)
        - Cost leverage (OpEx as % of revenue declining)

        Expansion Potential:
        - >2pp margin expansion: Strong upside    → 100pts
        - 1-2pp expansion: Moderate upside        → 75pts
        - 0-1pp: Margin pressure offsetting growth → 50pts
        - <0pp: Margin compression                → 25pts
        """

        rev_2021 = company.get("revenue_2021", 1)
        rev_2026 = company.get("revenue_2026", 1)
        oi_2021 = company.get("operating_income_2021", 0)
        oi_2026 = company.get("operating_income_2026", 0)
        ni_2021 = company.get("net_income_2021", 0)
        ni_2026 = company.get("net_income_2026", 0)

        oi_margin_2021 = (oi_2021 / (rev_2021 + 0.01)) * 100
        oi_margin_2026 = (oi_2026 / (rev_2026 + 0.01)) * 100
        oi_margin_delta = oi_margin_2026 - oi_margin_2021

        ni_margin_2021 = (ni_2021 / (rev_2021 + 0.01)) * 100
        ni_margin_2026 = (ni_2026 / (rev_2026 + 0.01)) * 100
        ni_margin_delta = ni_margin_2026 - ni_margin_2021

        # Average margin expansion
        avg_margin_delta = (oi_margin_delta + ni_margin_delta) / 2

        if avg_margin_delta > 2:
            score = 100
            potential = "Strong Expansion (>2pp)"
        elif avg_margin_delta > 1:
            score = 75
            potential = "Moderate Expansion (1-2pp)"
        elif avg_margin_delta > 0:
            score = 50
            potential = "Slight Expansion (0-1pp)"
        else:
            score = 25
            potential = "Margin Compression (<0pp)"

        return score, potential

    def analyze_competitive_positioning(self, company: Dict) -> Tuple[float, str]:
        """
        Signal 4: Competitive Positioning & Market Share

        Metrics:
        - Revenue growth vs sector average (relative strength)
        - Market cap to revenue ratio (scale/dominance)
        - Profitability rank (OI margin vs sector)

        Positioning:
        - Market leader (top revenue growth): Strong  → 100pts
        - Competitive (above-average growth): Healthy → 75pts
        - Follower (below-average growth): Moderate  → 50pts
        - Challenged (declining share): Weak         → 25pts
        """

        rev_2021 = company.get("revenue_2021", 1)
        rev_2026 = company.get("revenue_2026", 1)
        market_cap = company.get("market_cap_b", 1)

        revenue_growth = ((rev_2026 / rev_2021) ** (1 / 5) - 1) * 100
        _market_cap_to_rev = market_cap / (rev_2026 + 0.01)

        # Scoring: Growth trajectory
        if revenue_growth > 18:
            score = 100
            position = "Market Leader (>18% annual growth)"
        elif revenue_growth > 12:
            score = 75
            position = "Competitive (12-18% growth)"
        elif revenue_growth > 6:
            score = 50
            position = "Follower (6-12% growth)"
        else:
            score = 25
            position = "Challenged (<6% growth)"

        return score, position

    def calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """
        Combine 4 signals into single investment score (0-100)

        Weighting:
        - Capex ROI: 30% (most important for expansion thesis)
        - Debt paydown: 25% (financial health)
        - Margin expansion: 25% (value creation)
        - Competitive position: 20% (sustainability)
        """

        composite = (
            scores["capex_roi"] * 0.30
            + scores["debt_paydown"] * 0.25
            + scores["margin_expansion"] * 0.25
            + scores["competitive"] * 0.20
        )

        return round(composite, 1)

    def run_analysis(self) -> pd.DataFrame:
        """Analyze all Tier 2 candidates"""

        print("\n" + "=" * 80)
        print("SIMPLIFIED TIER 2 DEEP-DIVE ANALYSIS")
        print("=" * 80)
        print(f"\nAnalyzing {len(self.tier2)} Tier 2 candidates...")
        print("4 Investment Signals: Capex ROI | Debt Paydown | Margin Expansion | Position")

        results = []

        for idx, company in self.tier2.iterrows():
            company_dict = company.to_dict()

            # Signal 1: Capex ROI
            capex_roi_score, capex_phase = self.analyze_capex_roi(company_dict)

            # Signal 2: Debt Paydown
            debt_score, debt_timeline, paydown_yrs = self.analyze_debt_paydown(company_dict)

            # Signal 3: Margin Expansion
            margin_score, margin_potential = self.analyze_margin_expansion(company_dict)

            # Signal 4: Competitive Positioning
            comp_score, comp_position = self.analyze_competitive_positioning(company_dict)

            # Composite score
            scores = {
                "capex_roi": capex_roi_score,
                "debt_paydown": debt_score,
                "margin_expansion": margin_score,
                "competitive": comp_score,
            }
            composite = self.calculate_composite_score(scores)

            # Investment rating
            if composite >= 85:
                rating = "STRONG BUY"
            elif composite >= 75:
                rating = "BUY"
            elif composite >= 65:
                rating = "HOLD"
            else:
                rating = "WATCH"

            results.append(
                {
                    "ticker": company_dict.get("ticker", "N/A"),
                    "company": company_dict.get("company_name", "N/A"),
                    "sector": company_dict.get("sector", "N/A"),
                    "investment_score": composite,
                    "investment_rating": rating,
                    "capex_roi_score": capex_roi_score,
                    "capex_phase": capex_phase,
                    "debt_paydown_score": debt_score,
                    "debt_timeline": debt_timeline,
                    "paydown_years": round(paydown_yrs, 1),
                    "margin_score": margin_score,
                    "margin_potential": margin_potential,
                    "competitive_score": comp_score,
                    "competitive_position": comp_position,
                }
            )

        self.results_df = pd.DataFrame(results)
        return self.results_df

    def generate_reports(self) -> None:
        """Generate analysis reports"""

        if self.results_df.empty:
            print("No results to analyze")
            return

        print("\n" + "=" * 80)
        print("INVESTMENT RATING SUMMARY")
        print("=" * 80)

        rating_counts = self.results_df["investment_rating"].value_counts()
        for rating, count in rating_counts.items():
            pct = count / len(self.results_df) * 100
            print(f"\n{rating:15s}: {count:4d} companies ({pct:5.1f}%)")

        print("\n" + "=" * 80)
        print("TOP 20 BUY CANDIDATES (Highest Investment Score)")
        print("=" * 80)

        top_20 = self.results_df.nlargest(20, "investment_score")
        print("\n   Rank | Ticker | Company                | Score | Rating    | Debt Timeline")
        print("   " + "-" * 75)

        for rank, (_, row) in enumerate(top_20.iterrows(), 1):
            print(
                f"   {rank:2d}   | {row['ticker']:6s} | {row['company'][:20]:20s} | {row['investment_score']:5.1f} | {row['investment_rating']:9s} | {row['debt_timeline']}"
            )

        print("\n" + "=" * 80)
        print("SIGNAL BREAKDOWN ANALYSIS")
        print("=" * 80)

        print("\n📊 Average Scores Across 4 Signals:")
        print(f"   Capex ROI Score:         {self.results_df['capex_roi_score'].mean():.1f}/100")
        print(f"   Debt Paydown Score:      {self.results_df['debt_paydown_score'].mean():.1f}/100")
        print(f"   Margin Expansion Score:  {self.results_df['margin_score'].mean():.1f}/100")
        print(f"   Competitive Score:       {self.results_df['competitive_score'].mean():.1f}/100")

        print("\n🎯 Capex ROI Phase Distribution:")
        phase_dist = self.results_df["capex_phase"].value_counts()
        for phase, count in phase_dist.items():
            pct = count / len(self.results_df) * 100
            print(f"   {phase:40s}: {count:4d} ({pct:5.1f}%)")

        print("\n⏰ Debt Paydown Timeline Distribution:")
        timeline_dist = self.results_df["debt_timeline"].value_counts()
        for timeline, count in timeline_dist.items():
            pct = count / len(self.results_df) * 100
            print(f"   {timeline:35s}: {count:4d} ({pct:5.1f}%)")

        print("\n📈 Margin Expansion Potential Distribution:")
        margin_dist = self.results_df["margin_potential"].value_counts()
        for margin, count in margin_dist.items():
            pct = count / len(self.results_df) * 100
            print(f"   {margin:35s}: {count:4d} ({pct:5.1f}%)")

        print("\n" + "=" * 80)
        print("SECTOR ANALYSIS (By Average Investment Score)")
        print("=" * 80)

        sector_scores = (
            self.results_df.groupby("sector")["investment_score"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )

        print("\n   Top 10 Sectors by Investment Score:")
        print("   " + "-" * 70)
        for rank, (sector, row) in enumerate(sector_scores.head(10).iterrows(), 1):
            print(
                f"   {rank:2d}. {sector:30s} | Avg Score: {row['mean']:5.1f} | Companies: {int(row['count']):3d}"
            )

        print("\n" + "=" * 80)
        print("KEY INSIGHTS FOR PORTFOLIO MANAGERS")
        print("=" * 80)

        # 1. Best capex ROI opportunities
        best_roi = self.results_df.nlargest(5, "capex_roi_score")
        print("\n🔥 Best Capex ROI Candidates (Near-term Margin Expansion):")
        for _, row in best_roi.iterrows():
            print(
                f"   {row['ticker']:8s} | {row['capex_phase']:35s} | Score: {row['capex_roi_score']:.0f}"
            )

        # 2. Fastest debt paydown
        fastest_paydown = self.results_df.nsmallest(5, "paydown_years")
        print("\n💰 Fastest Debt Paydown (2026-2029):")
        for _, row in fastest_paydown.iterrows():
            print(
                f"   {row['ticker']:8s} | {row['debt_timeline']:35s} | {row['paydown_years']:4.1f} years"
            )

        # 3. Strongest competitive positions
        best_position = self.results_df.nlargest(5, "competitive_score")
        print("\n👑 Strongest Competitive Positions:")
        for _, row in best_position.iterrows():
            print(f"   {row['ticker']:8s} | {row['competitive_position']:40s}")

        # 4. Margin expansion plays
        best_margins = self.results_df.nlargest(5, "margin_score")
        print("\n📊 Best Margin Expansion Opportunities:")
        for _, row in best_margins.iterrows():
            print(f"   {row['ticker']:8s} | {row['margin_potential']:35s}")

        print("\n" + "=" * 80)
        print("INVESTMENT THESIS SUMMARY")
        print("=" * 80)

        strong_buy_count = len(
            self.results_df[self.results_df["investment_rating"] == "STRONG BUY"]
        )
        buy_count = len(self.results_df[self.results_df["investment_rating"] == "BUY"])

        print("\n✅ Actionable Recommendations:")
        print(
            f"   STRONG BUY (score ≥85):  {strong_buy_count:3d} candidates - Ready for investment"
        )
        print(f"   BUY (score 75-85):       {buy_count:3d} candidates - Review & shortlist")
        print("\n💡 Key Analysis Signals:")
        print("   1. Capex ROI (30% weight) - Profits accelerating relative to capex spend")
        print("   2. Debt Paydown (25%):    - Timeline to eliminate expansion debt")
        print("   3. Margin Expansion (25%): - Operating leverage from scale")
        print("   4. Competition (20%):     - Market share & growth sustainability")


if __name__ == "__main__":
    print("\n" + "🎯 " * 40)
    print("SIMPLIFIED TIER 2 ANALYSIS - DEEP DIVE FOR AGGRESSIVE EXPANDERS")
    print("🎯 " * 40)
