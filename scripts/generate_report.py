#!/usr/bin/env python3
"""
FluxHero Report Generator CLI

Command-line tool for generating performance reports and running Monte Carlo simulations.
Intended for quant research and offline analysis.

Usage:
    # Generate HTML tearsheet from sample data
    python scripts/generate_report.py --output my_report.html

    # Generate report with specific benchmark
    python scripts/generate_report.py --benchmark QQQ --output report.html

    # Run Monte Carlo simulation
    python scripts/generate_report.py --monte-carlo --simulations 10000

    # Show metrics only (no report generation)
    python scripts/generate_report.py --metrics-only

    # Full analysis with Monte Carlo
    python scripts/generate_report.py --full-analysis --output full_report.html

Reference: /Users/anvesh/.claude/plans/swirling-tumbling-cloud.md
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="FluxHero Report Generator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output report.html
  %(prog)s --benchmark QQQ --output qqq_comparison.html
  %(prog)s --monte-carlo --simulations 10000
  %(prog)s --metrics-only
  %(prog)s --full-analysis --output full_report.html
        """,
    )

    # Report generation options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for HTML report",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        default="SPY",
        help="Benchmark symbol for comparison (default: SPY)",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default="FluxHero Strategy Report",
        help="Report title",
    )

    # Data source options
    parser.add_argument(
        "--backtest",
        type=str,
        help="Backtest run ID to generate report from (requires database)",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        default=True,
        help="Use sample data for demonstration (default: True)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=252,
        help="Number of periods for sample data (default: 252)",
    )

    # Analysis options
    parser.add_argument(
        "--monte-carlo",
        "-mc",
        action="store_true",
        help="Run Monte Carlo simulation",
    )
    parser.add_argument(
        "--simulations",
        "-s",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument(
        "--mc-periods",
        type=int,
        default=252,
        help="Number of periods per Monte Carlo simulation (default: 252)",
    )

    # Output options
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Print metrics without generating report",
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Run full analysis including Monte Carlo and report generation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Import analytics modules (after path setup)
    try:
        from backend.analytics import (
            QuantStatsAdapter,
            TearsheetGenerator,
            MonteCarloSimulator,
            run_monte_carlo_analysis,
        )
    except ImportError as e:
        print(f"Error importing analytics modules: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)

    # Generate sample data
    print("Generating sample return data...")
    np.random.seed(42)

    # Realistic parameters
    daily_return = 0.0004  # ~10% annual
    daily_vol = 0.015  # ~24% annual
    n_periods = args.periods

    returns = np.random.normal(daily_return, daily_vol, n_periods)

    # Generate equity curve
    initial_capital = 100000.0
    equity_curve = np.zeros(n_periods + 1, dtype=np.float64)
    equity_curve[0] = initial_capital
    for i, r in enumerate(returns):
        equity_curve[i + 1] = equity_curve[i] * (1 + r)

    # Generate trade P&Ls
    n_trades = 50
    pnls = np.random.normal(200, 500, n_trades)

    print(f"Generated {n_periods} periods of data")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Final equity: ${equity_curve[-1]:,.2f}")
    print(f"Total return: {((equity_curve[-1] / initial_capital) - 1) * 100:.2f}%")
    print()

    # Create adapter and calculate metrics
    print("Calculating enhanced metrics...")
    adapter = QuantStatsAdapter(
        returns=returns,
        equity_curve=equity_curve,
        pnls=pnls,
        risk_free_rate=0.04,
    )

    tier1 = adapter.get_tier1_metrics()

    # Print Tier 1 metrics
    print("\n" + "=" * 60)
    print("TIER 1 METRICS (Numba-Optimized)")
    print("=" * 60)

    metric_groups = [
        ("Risk-Adjusted Returns", [
            ("Sortino Ratio", tier1["sortino_ratio"], "{:.2f}"),
            ("Calmar Ratio", tier1["calmar_ratio"], "{:.2f}"),
            ("Recovery Factor", tier1["recovery_factor"], "{:.2f}"),
        ]),
        ("Risk Metrics", [
            ("VaR (95%)", tier1["value_at_risk_95"] * 100, "{:.2f}%"),
            ("CVaR (95%)", tier1["cvar_95"] * 100, "{:.2f}%"),
            ("Ulcer Index", tier1["ulcer_index"], "{:.2f}"),
        ]),
        ("Trade Statistics", [
            ("Profit Factor", tier1["profit_factor"], "{:.2f}"),
            ("Kelly Criterion", tier1["kelly_criterion"] * 100, "{:.2f}%"),
            ("Max Consecutive Wins", tier1["max_consecutive_wins"], "{:d}"),
            ("Max Consecutive Losses", tier1["max_consecutive_losses"], "{:d}"),
        ]),
        ("Benchmark Comparison", [
            ("Alpha (annualized)", tier1["alpha"] * 100, "{:.2f}%"),
            ("Beta", tier1["beta"], "{:.2f}"),
            ("Information Ratio", tier1["information_ratio"], "{:.2f}"),
            ("R-Squared", tier1["r_squared"], "{:.2f}"),
        ]),
        ("Distribution", [
            ("Skewness", tier1["skewness"], "{:.2f}"),
            ("Kurtosis", tier1["kurtosis"], "{:.2f}"),
            ("Tail Ratio", tier1["tail_ratio"], "{:.2f}"),
        ]),
    ]

    for group_name, metrics in metric_groups:
        print(f"\n{group_name}")
        print("-" * 40)
        for name, value, fmt in metrics:
            if isinstance(value, int):
                formatted = fmt.format(value)
            else:
                formatted = fmt.format(float(value))
            print(f"  {name:<25} {formatted:>10}")

    # Monte Carlo analysis
    if args.monte_carlo or args.full_analysis:
        print("\n" + "=" * 60)
        print("MONTE CARLO SIMULATION")
        print("=" * 60)

        result = run_monte_carlo_analysis(
            returns=returns,
            n_simulations=args.simulations,
            n_periods=args.mc_periods,
            initial_capital=initial_capital,
            print_results=True,
        )

    # Generate report
    if args.output or args.full_analysis:
        output_path = args.output
        if args.full_analysis and not output_path:
            output_path = "fluxhero_full_analysis.html"

        print(f"\nGenerating HTML tearsheet report...")
        print(f"Benchmark: {args.benchmark}")

        generator = TearsheetGenerator()

        try:
            report_path = generator.generate_tearsheet(
                returns=returns,
                benchmark_symbol=args.benchmark,
                title=args.title,
                output_filename=output_path,
            )

            print(f"\nReport generated successfully!")
            print(f"Output: {report_path}")
            print(f"\nOpen in browser: file://{report_path.absolute()}")

        except Exception as e:
            print(f"\nError generating report: {e}")
            print("Trying without benchmark...")

            try:
                report_path = generator.generate_tearsheet(
                    returns=returns,
                    benchmark_symbol=None,
                    title=args.title,
                    output_filename=output_path,
                )
                print(f"Report generated (without benchmark): {report_path}")
            except Exception as e2:
                print(f"Failed to generate report: {e2}")
                sys.exit(1)

    elif args.metrics_only:
        print("\n(Metrics only mode - no report generated)")

    print("\nDone!")


if __name__ == "__main__":
    main()
