# FluxHero Enhancement Tasks

Each task is on a single line for ralphy.sh compatibility.

---

## Phase 18: Walk-Forward Testing (R9.4) - NOT IMPLEMENTED

- [x] Create walk-forward module (backend/backtesting/walk_forward.py) - implement WalkForwardWindow dataclass with train/test indices and dates, generate_walk_forward_windows() function with 3-month train (63 days) / 1-month test (21 days) windows, handle edge cases (insufficient data, uneven final window, date gaps). Ref: FLUXHERO_REQUIREMENTS.md R9.4.1

- [x] Implement rolling window execution (backend/backtesting/walk_forward.py) - create run_walk_forward_backtest() orchestrator that executes backtest on each test window sequentially, optionally re-optimizes strategy params on train window, tracks results per window. Ref: R9.4.2

- [x] Implement results aggregation (backend/backtesting/walk_forward.py) - create aggregate_walk_forward_results() to combine equity curves from all test periods, calculate aggregate Sharpe/drawdown/win_rate, count profitable windows. Ref: R9.4.3

- [x] Implement pass rate calculation (backend/backtesting/walk_forward.py) - calculate percentage of profitable test windows (final equity > initial), strategy passes if >60% profitable, add passes_walk_forward_test boolean. Ref: R9.4.4

- [x] Create walk-forward API endpoint (backend/api/server.py) - add WalkForwardRequest/WalkForwardResponse Pydantic models, create POST /api/backtest/walk-forward endpoint with per-window metrics in response.

- [ ] Add walk-forward frontend page (frontend/app/walk-forward/page.tsx) - create form for walk-forward config, display per-window results table, show pass/fail status, visualize equity curves per window.

- [ ] Create walk-forward unit tests (tests/unit/test_walk_forward.py) - test window generation with 12-month synthetic data, verify no data leakage between windows, test 4-month minimal case, test 1+ year multiple windows, test edge cases.

- [ ] Create walk-forward integration tests (tests/integration/test_walk_forward_backtest.py) - run walk-forward on 1-year SPY data, verify pass rate calculation, compare metrics against known values.

---

## Phase 19: Logging Enhancements - PARTIAL

- [ ] Add optional request body logging for development (backend/api/server.py) - add LOG_REQUEST_BODIES env flag, truncate bodies >500 chars, never log in production, mask sensitive fields (password, token, api_key).

- [ ] Add backtest operation logging (backend/backtesting/engine.py) - log backtest start with config summary, log progress every 10% of bars, log final metrics summary, include duration in milliseconds.

- [ ] Add strategy decision logging (backend/strategy/backtest_strategy.py) - add DEBUG level logging for signal generation, log regime changes, log entry/exit decisions with reasoning, configurable via log level.

- [ ] Evaluate structlog migration (backend/core/logging_config.py) - research structlog for better context propagation, compare with current setup, create decision document, pros: automatic context binding, better async support.

- [ ] Add audit logging for critical operations (backend/core/audit_logger.py) - create separate audit log file for trades, log all trade entries/exits with timestamps/prices/sizes/reasons, non-erasable for compliance.

---

## Phase 20: Return Calculations & Transformations - NEEDS LOG RETURNS

- [ ] Add log returns calculation (backend/backtesting/metrics.py) - implement calculate_log_returns() with formula np.log(equity[t]/equity[t-1]), more normally distributed than simple returns, better for statistical analysis.

- [ ] Add configurable return type (backend/backtesting/metrics.py) - add return_type parameter Literal["simple", "log"] defaulting to "simple", allow switching between methods, document trade-offs in docstring.

- [ ] Update Sharpe ratio for log returns (backend/backtesting/metrics.py lines 64-110) - add return_type parameter to calculate_sharpe_ratio(), formula works identically with either return type.

- [ ] Review annualization method (backend/backtesting/metrics.py lines 285-325) - current uses linear scaling (avg_return * 252), with log returns should sum then exp()-1, document i.i.d. assumption in calculate_annualized_return().

- [ ] Fix division by zero handling (backend/backtesting/metrics.py lines 55-58) - currently treats negative equity as 0 return which masks problems, should log warning and return NaN or raise exception.

- [ ] Add return transformation tests (tests/unit/test_metrics.py) - test simple vs log returns produce expected values, test with known equity curves, test edge cases (flat equity, negative equity).

---

## Phase 21: Trade Counting & Analysis - VERIFICATION

- [ ] Add trade counting assertion tests (tests/unit/test_trade_counting.py) - verify 5 entries + 5 exits = 5 trades, test partial fills counted correctly, test stop loss exit creates proper trade, test take profit exit creates proper trade.

- [ ] Verify win/loss classification (tests/unit/test_trade_counting.py) - test pnl>0 is win, test pnl=0 is loss (current behavior), test pnl<0 is loss, document breakeven handling decision.

- [ ] Add trade analytics (backend/backtesting/metrics.py) - calculate average holding period in bars, max consecutive wins/losses, profit factor (gross profit / gross loss), expectancy per trade.

- [ ] Add trade breakdown by regime (backend/backtesting/metrics.py) - separate win rate for trending vs mean-reverting regimes, track performance by strategy mode, identify which conditions work best.

---

## Phase 22: EMA & Indicator Verification - CORRECT IMPLEMENTATION

- [ ] Add EMA accuracy tests (tests/unit/test_indicators.py) - compare custom EMA vs pandas ewm(adjust=False), test with known price series, verify alpha calculation 2/(period+1), test periods 10/20/50.

- [ ] Add KAMA accuracy tests (tests/unit/test_adaptive_ema.py) - test efficiency ratio calculation, test adaptive smoothing constant, test KAMA vs known values, test regime classification thresholds (0.3/0.6).

- [ ] Document EMA decision (backend/computation/indicators.py) - add docstring explaining adjust=False choice, explain why custom implementation vs pandas, document Numba performance benefits.

- [ ] Add indicator warm-up validation (backend/strategy/backtest_strategy.py) - verify WARMUP_BARS=60 is sufficient (50-bar regression + 10-bar buffer), add assertion for minimum data length.

---

## Phase 23: Forward Bias Audit - PASSED

- [ ] Document no-future-signals rule (backend/backtesting/engine.py line 318) - add comment explaining i < n_bars-1 check prevents last bar signals, reference R9.1.4 no peeking into future.

- [ ] Add temporal ordering assertion (backend/backtesting/engine.py) - add runtime check signal_bar_index < fill_bar_index, use validate_no_lookahead() from fills.py, log warning if assertion fails.

- [ ] Document regime lag behavior (backend/strategy/regime_detector.py lines 482-536) - add docstring explaining 3-bar confirmation causes ~3 bar lag on market turns, note this is intentional to prevent whipsaws.

- [ ] Create forward bias checklist (docs/FORWARD_BIAS_CHECKLIST.md) - document all safeguards in place, create checklist for code review, include common bias patterns to avoid.

---

## Phase 24: Quality Control & Validation Framework - NEW

- [x] Create metric validation suite (tests/validation/test_metric_calculations.py) - use known equity curves with hand-calculated metrics, verify Sharpe/drawdown/win_rate match manual calculations, include worked examples in comments.

- [x] Create indicator validation suite (tests/validation/test_indicator_calculations.py) - use known price series with hand-calculated indicators, verify EMA(10) on [100,101,102...], verify RSI on overbought/oversold patterns, compare against TradingView.

- [x] Create signal validation suite (tests/validation/test_signal_generation.py) - verify trend-following signals on known trending data, verify mean-reversion signals on ranging data, test regime detection on synthetic transitions.

- [x] Add data validation on load (backend/data/yahoo_provider.py) - check for NaN in OHLCV, negative prices, volume=0, high<low errors, gaps>5 days missing data.

- [x] Add bar integrity checks (backend/backtesting/engine.py) - verify open/high/low/close relationships valid, verify timestamps monotonically increasing, log warnings for suspicious data.

- [x] Create golden test suite (tests/regression/test_golden_results.py) - run backtest on SPY 2020-2024 with fixed seed, store expected metrics in JSON, compare new runs against golden results, alert on >1% deviation.

- [x] Add benchmark comparison (tests/regression/test_benchmark_comparison.py) - compare strategy returns vs buy-and-hold, compare vs SPY total return, flag if significantly underperforming.

- [x] Create assumptions document (docs/ASSUMPTIONS.md) - document simple vs log returns choice, commission model ($0.005/share), slippage model (0.01%), fill assumptions (next-bar open), position sizing risk model.

- [ ] Add inline assumption comments (various files) - mark every assumption with # ASSUMPTION: comment, include rationale and alternatives considered, make assumptions searchable via grep.

- [x] Add sanity check assertions (backend/backtesting/engine.py) - assert equity never negative, position size <= max allowed, trades have valid entry < exit timestamps, P&L matches equity change.

- [x] Add metric sanity checks (backend/backtesting/metrics.py) - assert Sharpe in reasonable range (-5 to +5), win rate between 0 and 1, max drawdown <= 100%, log warning for extreme values.

---

## Priority Order

1. Phase 24: Quality Control - catches bugs before production
2. Phase 18: Walk-Forward Testing - critical for strategy validation
3. Phase 19: Logging - improves debuggability
4. Phase 20: Return Calculations - statistical improvement
5. Phase 21-23: Verification - lower priority, mostly documentation
