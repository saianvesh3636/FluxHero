"""
Cross-Asset Validator - Test Strategy Robustness Across Multiple Assets

Key Principle: If a parameter only works on one asset, it's likely overfit.

This module:
1. Tests calibrated parameters across multiple uncorrelated assets
2. Compares fixed vs calibrated threshold performance
3. Identifies asset-specific biases
4. Calculates robustness metrics

Usage:
    from backend.validation import CrossAssetValidator

    validator = CrossAssetValidator(
        assets=["SPY", "QQQ", "IWM", "GLD", "TLT", "AFRM"]
    )
    results = await validator.run_full_validation(bars_dict)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable
import numpy as np

from backend.calibration import (
    PercentileCalibrator,
    CalibratedParameters,
    WalkForwardCalibrator,
)
from backend.data import get_provider


@dataclass
class AssetValidationResult:
    """Validation results for a single asset."""
    symbol: str
    n_bars: int
    date_range: str

    # Calibrated parameters for this asset
    calibrated_params: CalibratedParameters

    # Performance metrics with FIXED thresholds
    fixed_sharpe: float = 0.0
    fixed_returns: float = 0.0
    fixed_max_drawdown: float = 0.0
    fixed_win_rate: float = 0.0
    fixed_n_trades: int = 0

    # Performance metrics with CALIBRATED thresholds
    calibrated_sharpe: float = 0.0
    calibrated_returns: float = 0.0
    calibrated_max_drawdown: float = 0.0
    calibrated_win_rate: float = 0.0
    calibrated_n_trades: int = 0

    # Improvement metrics
    sharpe_improvement: float = 0.0
    returns_improvement: float = 0.0

    # Parameter comparison with universal values
    rsi_oversold_vs_30: float = 0.0  # How different from magic number 30
    rsi_overbought_vs_70: float = 0.0
    er_trending_vs_06: float = 0.0


@dataclass
class ValidationResult:
    """Complete cross-asset validation results."""
    timestamp: str
    n_assets: int
    assets: list[str]

    # Individual asset results
    asset_results: list[AssetValidationResult] = field(default_factory=list)

    # Aggregate metrics
    avg_sharpe_fixed: float = 0.0
    avg_sharpe_calibrated: float = 0.0
    avg_improvement: float = 0.0

    # Robustness metrics
    sharpe_std_fixed: float = 0.0      # High std = asset-dependent
    sharpe_std_calibrated: float = 0.0  # Should be lower than fixed
    pct_assets_improved: float = 0.0    # % where calibrated > fixed

    # Parameter consistency
    rsi_oversold_range: tuple[float, float] = (0.0, 0.0)  # (min, max) across assets
    rsi_overbought_range: tuple[float, float] = (0.0, 0.0)
    param_variability_score: float = 0.0  # How much params vary across assets

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "n_assets": self.n_assets,
            "assets": self.assets,
            "avg_sharpe_fixed": self.avg_sharpe_fixed,
            "avg_sharpe_calibrated": self.avg_sharpe_calibrated,
            "avg_improvement": self.avg_improvement,
            "sharpe_std_fixed": self.sharpe_std_fixed,
            "sharpe_std_calibrated": self.sharpe_std_calibrated,
            "pct_assets_improved": self.pct_assets_improved,
            "rsi_oversold_range": self.rsi_oversold_range,
            "rsi_overbought_range": self.rsi_overbought_range,
        }


class CrossAssetValidator:
    """
    Validate strategy robustness across multiple assets.

    Tests whether calibrated parameters improve performance consistently
    across different asset types (stocks, ETFs, volatile vs stable).
    """

    # Default assets covering different characteristics
    DEFAULT_ASSETS = [
        "SPY",   # Large cap, low volatility
        "QQQ",   # Tech-heavy, medium volatility
        "IWM",   # Small cap, higher volatility
        "GLD",   # Commodity, uncorrelated to equities
        "TLT",   # Bonds, negative correlation to equities
    ]

    def __init__(
        self,
        assets: Optional[list[str]] = None,
        lookback_days: int = 365,
    ):
        """
        Initialize cross-asset validator.

        Parameters
        ----------
        assets : list[str], optional
            Assets to validate across. Defaults to SPY, QQQ, IWM, GLD, TLT.
        lookback_days : int
            Days of historical data to use.
        """
        self.assets = assets or self.DEFAULT_ASSETS
        self.lookback_days = lookback_days
        self.calibrator = PercentileCalibrator()

    async def fetch_data(self, symbol: str) -> tuple[np.ndarray, list[str]]:
        """Fetch historical data for a symbol."""
        provider = get_provider()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        data = await provider.fetch_historical_data(
            symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d"
        )

        return data.bars, data.dates

    def validate_single_asset(
        self,
        symbol: str,
        bars: np.ndarray,
        dates: list[str],
        evaluate_fn: Optional[Callable] = None
    ) -> AssetValidationResult:
        """
        Validate strategy on a single asset.

        Compares fixed vs calibrated performance.
        """
        # Calibrate parameters
        params = self.calibrator.calibrate(symbol, bars, dates)

        # Calculate differences from fixed thresholds
        rsi_oversold_diff = params.rsi_oversold - 30.0
        rsi_overbought_diff = params.rsi_overbought - 70.0
        er_trending_diff = params.er_trending - 0.6

        result = AssetValidationResult(
            symbol=symbol,
            n_bars=len(bars),
            date_range=f"{dates[0]} to {dates[-1]}",
            calibrated_params=params,
            rsi_oversold_vs_30=rsi_oversold_diff,
            rsi_overbought_vs_70=rsi_overbought_diff,
            er_trending_vs_06=er_trending_diff,
        )

        # Run backtests if evaluation function provided
        if evaluate_fn is not None:
            # Fixed thresholds backtest
            fixed_result = evaluate_fn(bars, None)  # None = use fixed
            result.fixed_sharpe = fixed_result.get("sharpe", 0)
            result.fixed_returns = fixed_result.get("total_return", 0)
            result.fixed_max_drawdown = fixed_result.get("max_drawdown", 0)
            result.fixed_win_rate = fixed_result.get("win_rate", 0)
            result.fixed_n_trades = fixed_result.get("n_trades", 0)

            # Calibrated thresholds backtest
            cal_result = evaluate_fn(bars, params)
            result.calibrated_sharpe = cal_result.get("sharpe", 0)
            result.calibrated_returns = cal_result.get("total_return", 0)
            result.calibrated_max_drawdown = cal_result.get("max_drawdown", 0)
            result.calibrated_win_rate = cal_result.get("win_rate", 0)
            result.calibrated_n_trades = cal_result.get("n_trades", 0)

            # Calculate improvements
            if result.fixed_sharpe != 0:
                result.sharpe_improvement = (
                    (result.calibrated_sharpe - result.fixed_sharpe) /
                    abs(result.fixed_sharpe) * 100
                )
            result.returns_improvement = result.calibrated_returns - result.fixed_returns

        return result

    async def run_full_validation(
        self,
        evaluate_fn: Optional[Callable] = None
    ) -> ValidationResult:
        """
        Run full cross-asset validation.

        Parameters
        ----------
        evaluate_fn : callable, optional
            Function to evaluate strategy: evaluate_fn(bars, params) -> dict

        Returns
        -------
        ValidationResult
            Complete validation results
        """
        result = ValidationResult(
            timestamp=datetime.now().isoformat(),
            n_assets=len(self.assets),
            assets=self.assets.copy(),
        )

        # Fetch data and validate each asset
        for symbol in self.assets:
            try:
                bars, dates = await self.fetch_data(symbol)
                asset_result = self.validate_single_asset(
                    symbol, bars, dates, evaluate_fn
                )
                result.asset_results.append(asset_result)
            except Exception as e:
                print(f"Error validating {symbol}: {e}")
                continue

        if not result.asset_results:
            return result

        # Calculate aggregate metrics
        sharpes_fixed = [r.fixed_sharpe for r in result.asset_results]
        sharpes_cal = [r.calibrated_sharpe for r in result.asset_results]
        rsi_os = [r.calibrated_params.rsi_oversold for r in result.asset_results]
        rsi_ob = [r.calibrated_params.rsi_overbought for r in result.asset_results]

        result.avg_sharpe_fixed = np.mean(sharpes_fixed)
        result.avg_sharpe_calibrated = np.mean(sharpes_cal)
        result.avg_improvement = np.mean([r.sharpe_improvement for r in result.asset_results])

        result.sharpe_std_fixed = np.std(sharpes_fixed)
        result.sharpe_std_calibrated = np.std(sharpes_cal)

        improved = sum(1 for r in result.asset_results if r.calibrated_sharpe > r.fixed_sharpe)
        result.pct_assets_improved = improved / len(result.asset_results) * 100

        result.rsi_oversold_range = (min(rsi_os), max(rsi_os))
        result.rsi_overbought_range = (min(rsi_ob), max(rsi_ob))

        # Parameter variability: higher = parameters differ more across assets
        # This is expected and shows adaptation, not a problem
        param_vars = [
            np.std(rsi_os) / 50,  # Normalized by typical range
            np.std(rsi_ob) / 50,
            np.std([r.calibrated_params.er_trending for r in result.asset_results]) / 0.5,
        ]
        result.param_variability_score = np.mean(param_vars)

        return result

    def print_validation_report(self, result: ValidationResult) -> None:
        """Print formatted validation report."""
        print(f"\n{'='*80}")
        print(f"  CROSS-ASSET VALIDATION REPORT")
        print(f"  Generated: {result.timestamp}")
        print(f"  Assets: {', '.join(result.assets)}")
        print(f"{'='*80}\n")

        # Individual asset results
        print(f"{'Asset':<8} {'RSI_OS':<10} {'RSI_OB':<10} {'Δ from 30':<12} {'Δ from 70':<12}")
        print("-"*60)
        for r in result.asset_results:
            print(f"{r.symbol:<8} {r.calibrated_params.rsi_oversold:<10.1f} "
                  f"{r.calibrated_params.rsi_overbought:<10.1f} "
                  f"{r.rsi_oversold_vs_30:>+10.1f} {r.rsi_overbought_vs_70:>+10.1f}")

        print(f"\n{'-'*60}")
        print(f"\n[AGGREGATE METRICS]")
        print(f"  RSI Oversold Range:  {result.rsi_oversold_range[0]:.1f} - {result.rsi_oversold_range[1]:.1f}")
        print(f"  RSI Overbought Range: {result.rsi_overbought_range[0]:.1f} - {result.rsi_overbought_range[1]:.1f}")
        print(f"  Parameter Variability: {result.param_variability_score:.3f}")

        if result.avg_sharpe_fixed != 0 or result.avg_sharpe_calibrated != 0:
            print(f"\n[PERFORMANCE COMPARISON]")
            print(f"  Avg Sharpe (Fixed):      {result.avg_sharpe_fixed:.3f}")
            print(f"  Avg Sharpe (Calibrated): {result.avg_sharpe_calibrated:.3f}")
            print(f"  Sharpe Std (Fixed):      {result.sharpe_std_fixed:.3f}")
            print(f"  Sharpe Std (Calibrated): {result.sharpe_std_calibrated:.3f}")
            print(f"  % Assets Improved:       {result.pct_assets_improved:.1f}%")

        print(f"\n{'='*80}")
        print(f"\n[KEY INSIGHT]")
        print(f"  RSI varies from {result.rsi_oversold_range[0]:.0f} to {result.rsi_oversold_range[1]:.0f}")
        print(f"  Using fixed '30' would miss this variation entirely.")
        print(f"  Each asset has its own natural RSI distribution.")
        print(f"\n{'='*80}\n")


async def run_quick_validation(assets: list[str] = None) -> ValidationResult:
    """
    Quick validation without backtesting - just compare parameter distributions.

    Useful for demonstrating why fixed thresholds fail.
    """
    validator = CrossAssetValidator(assets=assets)
    result = await validator.run_full_validation(evaluate_fn=None)
    validator.print_validation_report(result)
    return result
