#!/usr/bin/env python3
"""
Debt-for-Expansion & Profit-Reinvestment Analysis Tool
======================================================
Analyzes a company's 5-year financial data to identify:
1. Debt expansion phases (when company took on debt)
2. Capex acceleration (when capital deployment increased)
3. Profit reinvestment (when profits were retained vs distributed)
4. Sustainability (whether expansion is generating returns)

Usage:
    python analyze_debt_expansion.py

Data source: Download from SEC EDGAR or company investor relations
"""

from datetime import datetime

import pandas as pd


class DebtExpansionAnalyzer:
    """Analyzes debt-for-expansion and profit-reinvestment patterns"""

    def __init__(self, data_df, company_name, ticker):
        """
        Initialize analyzer with financial data

        Parameters:
        -----------
        data_df : pd.DataFrame
            Columns required:
            - fiscal_year, total_debt, total_equity, capex, revenue,
            - operating_cash_flow, operating_income, net_income,
            - retained_earnings, dividends_paid
        company_name : str
        ticker : str
        """
        self.data = data_df.copy()
        self.company_name = company_name
        self.ticker = ticker
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate all key financial ratios"""
        self.data["debt_to_equity"] = (self.data["total_debt"] / self.data["total_equity"]).round(3)

        self.data["capex_to_revenue"] = ((self.data["capex"] / self.data["revenue"]) * 100).round(1)

        self.data["capex_to_ocf"] = (
            (self.data["capex"] / self.data["operating_cash_flow"]) * 100
        ).round(1)

        self.data["net_capex"] = self.data["capex"] - self.data.get("depreciation", 0)

        self.data["free_cash_flow"] = (self.data["operating_cash_flow"] - self.data["capex"]).round(
            2
        )

        self.data["profit_retention_pct"] = (
            ((self.data["retained_earnings"].diff() / self.data["net_income"]) * 100).fillna(0)
        ).round(1)

        self.data["dividend_to_net_income"] = (
            (self.data["dividends_paid"] / self.data["net_income"]) * 100
        ).round(1)

        self.data["ocf_to_revenue"] = (
            (self.data["operating_cash_flow"] / self.data["revenue"]) * 100
        ).round(1)

    def identify_debt_phase(self):
        """Identify if company is in debt expansion phase"""
        print("\n" + "=" * 80)
        print(f"DEBT EXPANSION PHASE ANALYSIS: {self.company_name} ({self.ticker})")
        print("=" * 80)

        de_trend = self.data["debt_to_equity"].values
        years = self.data["fiscal_year"].values

        print("\nDebt-to-Equity Trend (5-year):")
        for year, de in zip(years, de_trend):
            print(f"  {year}: {de:.3f}", end="")
            if de > 0.5:
                print(" ⚠️ HIGH LEVERAGE")
            elif de > 0.3:
                print(" ✅ Moderate")
            else:
                print(" ✅ Low")

        # Calculate trend
        de_change = de_trend[-1] - de_trend[0]
        de_pct_change = (
            ((de_trend[-1] - de_trend[0]) / de_trend[0]) * 100 if de_trend[0] != 0 else 0
        )

        print(f"\n📊 Debt-to-Equity Change: {de_change:.3f} ({de_pct_change:+.1f}%)")

        if de_change > 0.05:
            print("   ✅ DEBT INCREASING (expansion phase)")
        elif de_change < -0.05:
            print("   ✅ DEBT DECREASING (paydown phase)")
        else:
            print("   → DEBT STABLE (maintaining position)")

        return de_change > 0.05  # True if expanding

    def identify_capex_phase(self):
        """Identify capex acceleration"""
        print("\n" + "=" * 80)
        print(f"CAPITAL EXPENDITURE ANALYSIS: {self.company_name}")
        print("=" * 80)

        capex_values = self.data["capex"].values
        capex_to_rev = self.data["capex_to_revenue"].values
        years = self.data["fiscal_year"].values

        print("\nCapital Expenditure Trend:")
        for year, capex, ratio in zip(years, capex_values, capex_to_rev):
            print(f"  {year}: ${capex:.1f}B ({ratio}% of revenue)", end="")
            if ratio > 25:
                print(" 🔴 EXTREME (>25%)")
            elif ratio > 15:
                print(" 🟠 AGGRESSIVE (15-25%)")
            elif ratio > 10:
                print(" 🟡 GROWTH (10-15%)")
            else:
                print(" 🟢 NORMAL (<10%)")

        # Capex acceleration
        capex_cagr = (
            (capex_values[-1] / capex_values[0]) ** (1 / (len(capex_values) - 1)) - 1
        ) * 100
        print(f"\n📊 Capex CAGR (5-year): {capex_cagr:+.1f}%")

        if capex_cagr > 20:
            print("   ✅ RAPID ACCELERATION (expansion phase)")
        elif capex_cagr > 5:
            print("   ✅ Moderate growth")
        else:
            print("   → Flat/declining (maintenance mode)")

        return capex_cagr > 20

    def identify_profit_reinvestment(self):
        """Identify profit reinvestment patterns"""
        print("\n" + "=" * 80)
        print(f"PROFIT REINVESTMENT ANALYSIS: {self.company_name}")
        print("=" * 80)

        re_values = self.data["retained_earnings"].values
        div_values = self.data["dividends_paid"].values
        ni_values = self.data["net_income"].values
        years = self.data["fiscal_year"].values

        print("\nRetained Earnings Trend:")
        for year, re, div, ni in zip(years, re_values, div_values, ni_values):
            print(
                f"  {year}: RE=${re:.1f}B | Dividends=${div:.1f}B ({(div/ni*100):.0f}% of income)"
            )

        # RE growth
        re_cagr = ((re_values[-1] / re_values[0]) ** (1 / (len(re_values) - 1)) - 1) * 100
        print(f"\n📊 Retained Earnings CAGR: {re_cagr:+.1f}%")

        # Dividend as % of income
        avg_div_pct = (div_values / ni_values).mean() * 100
        print(f"📊 Average Dividend Payout Ratio: {avg_div_pct:.1f}%")

        if re_cagr > 10:
            print("   ✅ STRONG REINVESTMENT (profits retained)")
        elif re_cagr > 5:
            print("   ✅ Moderate reinvestment")
        else:
            print("   → Minimal reinvestment")

        if avg_div_pct < 30:
            print("   ✅ PROFIT RETENTION HIGH (<30% dividends)")
        elif avg_div_pct < 50:
            print("   ✅ Moderate profit retention (30-50% dividends)")
        else:
            print("   → High dividend payout (>50% profits distributed)")

        return re_cagr > 10 and avg_div_pct < 50

    def identify_sustainability(self):
        """Check if expansion is sustainable (capex ROI)"""
        print("\n" + "=" * 80)
        print(f"SUSTAINABILITY CHECK: {self.company_name}")
        print("=" * 80)

        capex_to_ocf = self.data["capex_to_ocf"].values
        fcf = self.data["free_cash_flow"].values
        _oi = self.data["operating_income"].values
        years = self.data["fiscal_year"].values

        print("\nCapex-to-Operating-Cash-Flow (% of OCF reinvested):")
        for year, ratio, cash_flow in zip(years, capex_to_ocf, fcf):
            print(f"  {year}: {ratio:.0f}% (FCF: ${cash_flow:.1f}B)", end="")
            if ratio > 100:
                print(" 🔴 UNSUSTAINABLE (>100% OCF)")
            elif ratio > 80:
                print(" 🟠 HIGH (80-100%)")
            elif ratio > 50:
                print(" 🟡 MODERATE (50-80%)")
            else:
                print(" 🟢 SUSTAINABLE (<50%)")

        # Check if OCF declining while capex rising
        ocf_trend = self.data["operating_cash_flow"].values
        capex_trend = self.data["capex"].values

        ocf_declining = ocf_trend[-1] < ocf_trend[0]
        capex_rising = capex_trend[-1] > capex_trend[0]

        print("\n⚠️ Warning Signs:")
        if ocf_declining and capex_rising:
            print("   🔴 OCF declining while capex rising = UNSUSTAINABLE")
            sustainability_score = 1
        elif any(f < 0 for f in fcf):
            print("   🟠 Negative FCF in some years")
            sustainability_score = 2
        elif capex_to_ocf.mean() > 80:
            print("   🟡 High capex-to-OCF ratio (capex strains cash flow)")
            sustainability_score = 3
        else:
            print("   ✅ Healthy cash flow (expansion sustainable)")
            sustainability_score = 4

        return sustainability_score >= 3

    def calculate_debt_expansion_score(self):
        """
        Calculate overall score for debt-for-expansion + reinvestment
        Returns: score 1-5 (1=weak, 5=strong)
        """
        print("\n" + "=" * 80)
        print(f"OVERALL DEBT-EXPANSION SCORE: {self.company_name}")
        print("=" * 80)

        score = 0
        checks_passed = 0
        total_checks = 6

        # Check 1: Debt increasing
        de_change = self.data["debt_to_equity"].iloc[-1] - self.data["debt_to_equity"].iloc[0]
        if de_change > 0.05:
            score += 1
            checks_passed += 1
            print("✅ Check 1: Debt-to-Equity rising (debt expansion)")
        else:
            print("❌ Check 1: Debt-to-Equity not rising")

        # Check 2: Capex accelerating
        capex_values = self.data["capex"].values
        capex_cagr = (
            (capex_values[-1] / capex_values[0]) ** (1 / (len(capex_values) - 1)) - 1
        ) * 100
        if capex_cagr > 15:
            score += 1
            checks_passed += 1
            print("✅ Check 2: Capex accelerating (15%+ CAGR)")
        else:
            print("❌ Check 2: Capex growth <15% CAGR")

        # Check 3: Capex/Revenue high
        if self.data["capex_to_revenue"].iloc[-1] > 10:
            score += 1
            checks_passed += 1
            print("✅ Check 3: Capex-to-Revenue >10% (expansion phase)")
        else:
            print("❌ Check 3: Capex-to-Revenue <10%")

        # Check 4: Profit reinvestment
        re_values = self.data["retained_earnings"].values
        re_cagr = ((re_values[-1] / re_values[0]) ** (1 / (len(re_values) - 1)) - 1) * 100
        if re_cagr > 8:
            score += 1
            checks_passed += 1
            print("✅ Check 4: Retained earnings growing 8%+ (profit reinvested)")
        else:
            print("❌ Check 4: Retained earnings growth <8%")

        # Check 5: Dividend not rising
        div_values = self.data["dividends_paid"].values
        div_change = ((div_values[-1] - div_values[0]) / (div_values[0] + 0.01)) * 100
        if div_change < 20:  # Dividends relatively flat
            score += 1
            checks_passed += 1
            print("✅ Check 5: Dividends stable/declining (profits retained)")
        else:
            print("❌ Check 5: Dividends rising faster than profit growth")

        # Check 6: Sustainable FCF
        fcf = self.data["free_cash_flow"].values
        if fcf.mean() > 0 and fcf[-1] > 0:
            score += 1
            checks_passed += 1
            print("✅ Check 6: Positive FCF (expansion sustainable)")
        elif fcf.mean() > -5:  # Slightly negative OK if temporary
            checks_passed += 0.5
            print("⚠️  Check 6: Near-zero FCF (tolerable for high-growth)")
        else:
            print("❌ Check 6: Negative FCF (unsustainable)")

        print(f"\n📊 SCORE: {score:.1f} / 5.0 checks passed")
        print(f"   Checks Passed: {checks_passed} / {total_checks}")

        if score >= 4.5:
            rating = "⭐⭐⭐⭐⭐ TEXTBOOK CASE - Strong debt-for-expansion + reinvestment"
        elif score >= 3.5:
            rating = "⭐⭐⭐⭐ STRONG - Clear expansion pattern"
        elif score >= 2.5:
            rating = "⭐⭐⭐ MODERATE - Some expansion signals"
        elif score >= 1.5:
            rating = "⭐⭐ WEAK - Limited expansion pattern"
        else:
            rating = "⭐ NO PATTERN - Not expanding or reinvesting"

        print(f"   Rating: {rating}\n")

        return score

    def print_summary_table(self):
        """Print summary table of all metrics"""
        print("\n" + "=" * 80)
        print(f"SUMMARY TABLE: {self.company_name} ({self.ticker})")
        print("=" * 80 + "\n")

        summary = self.data[
            [
                "fiscal_year",
                "debt_to_equity",
                "capex",
                "capex_to_revenue",
                "capex_to_ocf",
                "free_cash_flow",
                "retained_earnings",
                "dividends_paid",
            ]
        ].copy()

        summary.columns = [
            "Year",
            "D/E",
            "Capex($B)",
            "Capex/Rev%",
            "Capex/OCF%",
            "FCF($B)",
            "Retained($B)",
            "Div($B)",
        ]

        print(summary.to_string(index=False))
        print()

    def generate_report(self):
        """Generate complete analysis report"""
        print("\n\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " DEBT-FOR-EXPANSION & PROFIT-REINVESTMENT ANALYSIS REPORT".center(78) + "║")
        print("║" + f" {self.company_name} ({self.ticker})".center(78) + "║")
        print("║" + f" Generated: {datetime.now().strftime('%B %d, %Y')}".center(78) + "║")
        print("╚" + "=" * 78 + "╝")

        # Run all analyses
        debt_expanding = self.identify_debt_phase()
        capex_accelerating = self.identify_capex_phase()
        profit_reinvesting = self.identify_profit_reinvestment()
        sustainable = self.identify_sustainability()
        _overall_score = self.calculate_debt_expansion_score()

        # Print summary table
        self.print_summary_table()

        # Print recommendation
        print("\n" + "=" * 80)
        print("RECOMMENDATION:")
        print("=" * 80)

        if debt_expanding and capex_accelerating and profit_reinvesting:
            print(
                """
✅ STRONG DEBT-FOR-EXPANSION PATTERN DETECTED

This company is:
1. ✅ Taking on debt (D/E ratio rising)
2. ✅ Deploying capital at scale (capex accelerating)
3. ✅ Reinvesting profits (retained earnings growing, dividends flat)
4. ✅ Financing sustainably (positive or near-neutral FCF)

THESIS: Company is in strategic expansion phase, using a combination of
debt + retained profits to fund growth capex. Watch for:
- ROI on new capacity (should translate to higher operating income in 2-3 years)
- Debt paydown as capex phase ends (D/E should decline post-2027)
- Profit growth acceleration once capex ramps mature

INVESTOR ACTION: Monitor profitability trends and capex ROI metrics
            """
            )

        elif debt_expanding and capex_accelerating and not sustainable:
            print(
                """
⚠️  DEBT-EXPANSION BUT QUESTIONABLE SUSTAINABILITY

This company is taking on debt and increasing capex, BUT:
- Free cash flow is negative or deteriorating
- OCF not growing as fast as capex
- Relying on continuous debt financing

RISK: If capex ROI disappoints or demand softens, company may need
      restructuring/dividend cuts. Examples: Intel (2024-2025)

INVESTOR ACTION: Proceed with caution; watch quarterly FCF trends closely
            """
            )

        elif not debt_expanding and capex_accelerating:
            print(
                """
💡 CAPEX ACCELERATION WITHOUT DEBT INCREASE

This company is growing capex WITHOUT taking on more debt:
- Using operating cash flow + retained earnings
- Conservative debt management
- Healthy self-funding model

Examples: Microsoft, Google (though they did issue some strategic debt)

INVESTOR ACTION: Generally positive signal; no leverage concerns
            """
            )

        else:
            print(
                """
❌ NO STRONG DEBT-FOR-EXPANSION PATTERN

This company is NOT in an aggressive expansion phase:
- Debt stable or declining
- Capex flat or modest
- Profits distributed (dividends/buybacks) rather than reinvested

Examples: ASML (asset-light model), mature tech companies

INVESTOR ACTION: Company likely in steady-state; expect modest growth
            """
            )

        print("\n" + "=" * 80)


# Example usage
if __name__ == "__main__":

    # Example data: Micron Technology
    micron_data = {
        "fiscal_year": [2021, 2022, 2023, 2024, 2025, 2026],
        "total_debt": [5.2, 6.1, 9.3, 7.8, 4.6, 4.2],
        "total_equity": [20.8, 21.2, 26.5, 30.2, 30.9, 28.2],
        "capex": [5.8, 6.5, 7.2, 12.4, 15.9, 25.0],
        "revenue": [27.4, 30.2, 30.8, 32.1, 36.5, 39.2],
        "operating_cash_flow": [14.2, 11.8, 12.5, 18.2, 22.1, 28.5],
        "operating_income": [8.5, 6.2, 4.1, 12.8, 17.4, 21.5],
        "net_income": [6.3, 1.5, -2.1, 9.2, 13.2, 17.5],
        "retained_earnings": [34.7, 36.8, 52.1, 73.5, 78.4, 94.7],
        "dividends_paid": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    }

    df = pd.DataFrame(micron_data)

    analyzer = DebtExpansionAnalyzer(df, "Micron Technology", "MU")
    analyzer.generate_report()

    print("\n\n📌 To analyze another company:")
    print("   1. Create a DataFrame with the same columns")
    print("   2. Call: analyzer = DebtExpansionAnalyzer(df, 'Company Name', 'TICKER')")
    print("   3. Call: analyzer.generate_report()")
