"""
Parameter Store - Data Classes for Calibrated Parameters

All parameters are derived from data, not magic numbers.
Uses percentile-based thresholds that adapt to each asset's distribution.

Usage:
    from backend.calibration import CalibratedParameters, ParameterStore

    # Load or create parameters
    store = ParameterStore()
    params = store.get_or_calibrate("SPY", bars)
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json
from pathlib import Path
import numpy as np


@dataclass
class IndicatorDistribution:
    """Statistics for an indicator's distribution over a lookback period."""

    mean: float
    std: float
    min: float
    max: float
    p5: float    # 5th percentile
    p10: float   # 10th percentile
    p25: float   # 25th percentile (Q1)
    p50: float   # 50th percentile (median)
    p75: float   # 75th percentile (Q3)
    p90: float   # 90th percentile
    p95: float   # 95th percentile
    n_samples: int

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "IndicatorDistribution":
        """Calculate distribution statistics from array."""
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return cls(
                mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
                p5=np.nan, p10=np.nan, p25=np.nan, p50=np.nan,
                p75=np.nan, p90=np.nan, p95=np.nan, n_samples=0
            )

        return cls(
            mean=float(np.mean(valid)),
            std=float(np.std(valid)),
            min=float(np.min(valid)),
            max=float(np.max(valid)),
            p5=float(np.percentile(valid, 5)),
            p10=float(np.percentile(valid, 10)),
            p25=float(np.percentile(valid, 25)),
            p50=float(np.percentile(valid, 50)),
            p75=float(np.percentile(valid, 75)),
            p90=float(np.percentile(valid, 90)),
            p95=float(np.percentile(valid, 95)),
            n_samples=len(valid)
        )


@dataclass
class CalibratedParameters:
    """
    Fully calibrated parameters for a specific asset.

    All thresholds are derived from percentiles of the asset's own distribution.
    No magic numbers - everything is data-driven.
    """

    # Metadata
    symbol: str
    calibration_date: str
    lookback_bars: int
    data_start_date: str
    data_end_date: str

    # RSI Thresholds (from RSI distribution)
    rsi_oversold: float         # 10th percentile - dynamic oversold level
    rsi_overbought: float       # 90th percentile - dynamic overbought level
    rsi_extreme_oversold: float # 5th percentile - extreme oversold
    rsi_extreme_overbought: float  # 95th percentile - extreme overbought
    rsi_neutral_low: float      # 25th percentile
    rsi_neutral_high: float     # 75th percentile

    # Efficiency Ratio Thresholds (from ER distribution)
    er_choppy: float            # 25th percentile - choppy/ranging threshold
    er_trending: float          # 75th percentile - trending threshold
    er_strong_trend: float      # 90th percentile - strong trend

    # Volatility Thresholds (from ATR/ATR_MA ratio distribution)
    vol_low: float              # 20th percentile - low volatility
    vol_normal_low: float       # 40th percentile
    vol_normal_high: float      # 60th percentile
    vol_high: float             # 80th percentile - high volatility
    vol_extreme: float          # 95th percentile - extreme volatility

    # Golden Alpha Thresholds (from alpha distribution)
    alpha_slow_regime: float    # 25th percentile - mean-reversion regime
    alpha_fast_regime: float    # 75th percentile - trending regime

    # ATR-based multipliers (calibrated from price movement analysis)
    atr_entry_multiplier: float     # Entry band width
    atr_exit_multiplier: float      # Exit band width
    atr_stop_multiplier: float      # Stop loss distance

    # Distribution objects for reference
    rsi_distribution: Optional[IndicatorDistribution] = None
    er_distribution: Optional[IndicatorDistribution] = None
    vol_ratio_distribution: Optional[IndicatorDistribution] = None
    alpha_distribution: Optional[IndicatorDistribution] = None

    # Validation metrics
    sharpe_in_sample: Optional[float] = None
    parameter_stability: Optional[float] = None  # How much params changed vs prior

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert distribution objects to dicts
        for key in ['rsi_distribution', 'er_distribution', 'vol_ratio_distribution', 'alpha_distribution']:
            if d[key] is not None:
                d[key] = asdict(d[key])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalibratedParameters":
        """Create from dictionary."""
        # Convert distribution dicts back to objects
        for key in ['rsi_distribution', 'er_distribution', 'vol_ratio_distribution', 'alpha_distribution']:
            if d.get(key) is not None:
                d[key] = IndicatorDistribution(**d[key])
        return cls(**d)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CalibratedParameters":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class CalibrationConfig:
    """Configuration for the calibration process."""

    # Lookback periods
    lookback_bars: int = 252        # 1 year of daily bars
    min_bars_required: int = 100    # Minimum bars for valid calibration

    # Recalibration
    recalibrate_every_bars: int = 21  # Monthly recalibration

    # Percentile settings (these are NOT magic numbers - they define what "extreme" means statistically)
    oversold_percentile: float = 10.0
    overbought_percentile: float = 90.0
    extreme_percentile: float = 5.0
    quartile_low: float = 25.0
    quartile_high: float = 75.0

    # ATR multiplier estimation settings
    atr_lookback_for_multipliers: int = 50  # Bars to analyze for ATR calibration


class ParameterStore:
    """
    Store and retrieve calibrated parameters.

    Supports:
    - In-memory caching
    - JSON file persistence
    - Automatic staleness detection
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("data/calibrations")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, CalibratedParameters] = {}

    def _get_file_path(self, symbol: str) -> Path:
        """Get storage file path for a symbol."""
        return self.storage_dir / f"{symbol.upper()}_calibration.json"

    def save(self, params: CalibratedParameters) -> None:
        """Save calibrated parameters to file and cache."""
        self._cache[params.symbol.upper()] = params

        file_path = self._get_file_path(params.symbol)
        with open(file_path, 'w') as f:
            f.write(params.to_json())

    def load(self, symbol: str) -> Optional[CalibratedParameters]:
        """Load calibrated parameters from cache or file."""
        symbol = symbol.upper()

        # Check cache first
        if symbol in self._cache:
            return self._cache[symbol]

        # Try loading from file
        file_path = self._get_file_path(symbol)
        if file_path.exists():
            with open(file_path, 'r') as f:
                params = CalibratedParameters.from_json(f.read())
                self._cache[symbol] = params
                return params

        return None

    def is_stale(
        self,
        symbol: str,
        current_bar: int,
        last_calibration_bar: int,
        recalibrate_every: int = 21
    ) -> bool:
        """Check if parameters need recalibration."""
        return (current_bar - last_calibration_bar) >= recalibrate_every

    def is_date_stale(
        self,
        symbol: str,
        max_age_days: int = 30
    ) -> bool:
        """Check if parameters are stale based on calibration date."""
        params = self.load(symbol)
        if params is None:
            return True

        cal_date = datetime.strptime(params.calibration_date, "%Y-%m-%d")
        age = (datetime.now() - cal_date).days
        return age > max_age_days

    def list_symbols(self) -> list[str]:
        """List all symbols with stored calibrations."""
        return [f.stem.replace("_calibration", "")
                for f in self.storage_dir.glob("*_calibration.json")]

    def delete(self, symbol: str) -> bool:
        """Delete calibration for a symbol."""
        symbol = symbol.upper()

        if symbol in self._cache:
            del self._cache[symbol]

        file_path = self._get_file_path(symbol)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
