#!/usr/bin/env python3
"""
Fetch 5-Year Financial Data for Stock Universe
===============================================
Pulls financial data from Yahoo Finance API for screening.
Supports US stocks, international stocks, and custom watchlists.

Usage:
    python fetch_stock_data.py --source "us_large_cap" --output data.csv
    python fetch_stock_data.py --tickers "MU,INTC,MSFT,GOOGL" --output my_stocks.csv
"""

import sys

import pandas as pd
import yfinance as yf


class StockDataFetcher:
    """Fetch 5-year financial data for stock screening"""

    # Predefined stock universes
    UNIVERSES = {
        "us_mega_cap": [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "BRK.B",
            "JNJ",
            "JPM",
        ],
        "us_large_cap_tech": [
            "INTC",
            "AMD",
            "QCOM",
            "CSCO",
            "CRM",
            "ADBE",
            "AVGO",
            "AMAT",
            "LRCX",
            "MRVL",
        ],
        "us_semiconductor": [
            "MU",
            "INTC",
            "NVDA",
            "AMD",
            "QCOM",
            "AVGO",
            "LRCX",
            "AMAT",
            "MRVL",
            "ON",
        ],
        "us_cloud_infra": [
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "CRM",
            "IBM",
            "ORCL",
            "ACHR",
            "DDOG",
            "CRWD",
        ],
        "us_ev_automotive": [
            "TSLA",
            "F",
            "GM",
            "LCID",
            "RIVN",
            "XPEV",
            "NIO",
            "LI",
            "BLNK",
            "PLUG",
        ],
        "us_renewable_energy": [
            "NEE",
            "ENPH",
            "SEDG",
            "ICLN",
            "TAN",
            "RUN",
            "PLUG",
            "FSLR",
            "ADTN",
            "DKNG",
        ],
        "us_telecom": ["VZ", "T", "TMUS", "CMCSA", "CHTR", "LYV", "DISH", "LUMN", "SHENX", "TDS"],
        "us_industrials": ["GE", "BA", "CAT", "PCAR", "LMT", "RTX", "HOG", "EMR", "LUV", "DAL"],
        "us_pharmaceuticals": [
            "JNJ",
            "PFE",
            "ABBV",
            "MRK",
            "LLY",
            "BMY",
            "TMDX",
            "REGN",
            "AMGN",
            "GILD",
        ],
        "us_retail": ["WMT", "AMZN", "HD", "NKE", "MCD", "SBUX", "TJX", "LULU", "DECK", "ORLY"],
        "us_financials": ["JPM", "BAC", "WFC", "GS", "BLK", "COIN", "KKR", "APO", "BX", "SCHW"],
    }

    def __init__(self):
        self.data = pd.DataFrame()
        self.failed_tickers = []

    def fetch_ticker_data(self, ticker, years=5):
        """Fetch 5-year financial data for single ticker"""
        try:
            print(f"  Fetching {ticker}...", end=" ", flush=True)

            # Create ticker object
            t = yf.Ticker(ticker)

            # Get basic info
            info = t.info if hasattr(t, "info") else {}

            # Get financials (4 years available from yfinance)
            try:
                balance_sheet = t.balance_sheet
                income_stmt = t.income_stmt
                cash_flow = t.cash_flow
            except Exception:
                print("❌ (no financials)")
                return None

            # Get historical price for market cap (latest)
            try:
                hist = t.history(period="1d")
                if hist.empty:
                    print("❌ (no price data)")
                    return None
            except Exception:
                print("❌ (no price data)")
                return None

            # Extract key metrics from balance sheet, income statement, cash flow
            # Note: yfinance provides quarterly and annual data

            record = {
                "ticker": ticker,
                "company_name": info.get("longName", ticker),
                "sector": info.get("sector", "Unknown"),
                "market_cap_b": info.get("marketCap", 0) / 1e9 if info.get("marketCap") else 0,
                "country": info.get("country", "US"),
            }

            # Extract financials for available years (typically last 4)
            _years_available = []
            if not balance_sheet.empty:
                _years_available = balance_sheet.columns.year.unique()[:4]

            # Populate data with None for missing years, actual values for available
            for year in range(2022, 2027):  # 2022-2026
                # Try to get data for this year
                if not balance_sheet.empty:
                    bs_year = (
                        balance_sheet.xs(year, level="year", axis=1)
                        if year in balance_sheet.columns.year
                        else None
                    )
                else:
                    bs_year = None

                if not income_stmt.empty:
                    is_year = (
                        income_stmt.xs(year, level="year", axis=1)
                        if year in income_stmt.columns.year
                        else None
                    )
                else:
                    is_year = None

                if not cash_flow.empty:
                    cf_year = (
                        cash_flow.xs(year, level="year", axis=1)
                        if year in cash_flow.columns.year
                        else None
                    )
                else:
                    cf_year = None

                # Extract specific metrics
                try:
                    # Balance sheet items
                    total_debt = (bs_year.get("TotalDebt") if bs_year is not None else None) or 0
                    total_equity = (
                        bs_year.get("StockholderEquity") if bs_year is not None else None
                    ) or 1

                    # Income statement items
                    revenue = (is_year.get("TotalRevenue") if is_year is not None else None) or 1
                    operating_income = (
                        is_year.get("OperatingIncome") if is_year is not None else None
                    ) or 0
                    net_income = (is_year.get("NetIncome") if is_year is not None else None) or 0

                    # Cash flow items
                    operating_cf = (
                        cf_year.get("OperatingCashFlow") if cf_year is not None else None
                    ) or 0
                    capex = abs(
                        (cf_year.get("CapitalExpenditures") if cf_year is not None else None) or 0
                    )

                    # Estimate retained earnings (simplified)
                    retained_earnings = (
                        bs_year.get("RetainedEarnings") if bs_year is not None else None
                    ) or 0

                except Exception:
                    total_debt, total_equity = 0, 1
                    revenue, operating_income, net_income = 1, 0, 0
                    operating_cf, capex = 0, 0
                    retained_earnings = 0

                # Store with year suffix
                record[f"total_debt_{year}"] = total_debt / 1e9 if total_debt else 0
                record[f"total_equity_{year}"] = total_equity / 1e9 if total_equity else 1
                record[f"capex_{year}"] = capex / 1e9 if capex else 0
                record[f"revenue_{year}"] = revenue / 1e9 if revenue else 1
                record[f"operating_cf_{year}"] = operating_cf / 1e9 if operating_cf else 0
                record[f"operating_income_{year}"] = (
                    operating_income / 1e9 if operating_income else 0
                )
                record[f"net_income_{year}"] = net_income / 1e9 if net_income else 0
                record[f"retained_earnings_{year}"] = (
                    retained_earnings / 1e9 if retained_earnings else 0
                )

            # Dividends (estimate from cash flow or use 0)
            for year in range(2022, 2027):
                record[f"dividends_paid_{year}"] = 0  # Would need separate API call

            print("✅")
            return record

        except Exception as e:
            print(f"❌ ({str(e)[:20]})")
            self.failed_tickers.append((ticker, str(e)))
            return None

    def fetch_universe(self, universe_name):
        """Fetch data for predefined universe"""
        if universe_name not in self.UNIVERSES:
            print(f"❌ Universe '{universe_name}' not found")
            print(f"   Available: {', '.join(self.UNIVERSES.keys())}")
            return None

        tickers = self.UNIVERSES[universe_name]
        print(f"\nFetching {len(tickers)} stocks from '{universe_name}' universe...")
        return self.fetch_tickers(tickers)

    def fetch_tickers(self, tickers):
        """Fetch data for list of tickers"""
        records = []

        print(f"\nFetching data for {len(tickers)} tickers...")
        for ticker in tickers:
            record = self.fetch_ticker_data(ticker.strip())
            if record:
                records.append(record)

        self.data = pd.DataFrame(records)

        print(f"\n✅ Successfully fetched: {len(records)}/{len(tickers)} tickers")
        if self.failed_tickers:
            print(f"❌ Failed: {len(self.failed_tickers)} tickers")
            for ticker, reason in self.failed_tickers[:5]:
                print(f"   - {ticker}: {reason}")

        return self.data

    def save_csv(self, output_path):
        """Save to CSV for screening"""
        if self.data.empty:
            print("❌ No data to save")
            return False

        self.data.to_csv(output_path, index=False)
        print(f"\n✅ Saved to {output_path}")
        print(f"   Rows: {len(self.data)}")
        print(f"   Columns: {len(self.data.columns)}")

        return True

    def print_summary(self):
        """Print data summary"""
        if self.data.empty:
            print("No data available")
            return

        print("\n" + "=" * 80)
        print("DATA SUMMARY")
        print("=" * 80)
        print(f"Total companies: {len(self.data)}")
        print("\nBy Sector:")
        print(self.data["sector"].value_counts().head(10))
        print("\nBy Country:")
        print(self.data["country"].value_counts().head(10))
        print("\nMarket Cap Distribution (B):")
        print(self.data["market_cap_b"].describe())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch stock financial data for screening")
    parser.add_argument("--universe", type=str, help="Predefined universe name")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers")
    parser.add_argument(
        "--output", type=str, default="stock_financial_data.csv", help="Output CSV file"
    )

    args = parser.parse_args()

    fetcher = StockDataFetcher()

    if args.universe:
        fetcher.fetch_universe(args.universe)
    elif args.tickers:
        tickers = args.tickers.split(",")
        fetcher.fetch_tickers(tickers)
    else:
        # Default: fetch mega-cap tech stocks
        print("No universe or tickers specified. Fetching us_mega_cap universe...")
        fetcher.fetch_universe("us_mega_cap")

    if not fetcher.data.empty:
        fetcher.save_csv(args.output)
        fetcher.print_summary()
    else:
        print("❌ No data fetched")
        sys.exit(1)
