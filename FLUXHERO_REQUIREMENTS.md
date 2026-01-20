# FluxHero: Retail Quant Trading System
## Requirements & Success Criteria

**Version**: 1.0
**Project Name**: FluxHero (Flux = constant adaptation, Hero = mastery)
**Target User**: Solo retail developer
**Market**: US Equities (stocks/ETFs)
**Philosophy**: High-performance Python, adaptive strategies, lightweight infrastructure

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Feature 1: JIT Computation Engine](#feature-1-jit-computation-engine)
3. [Feature 2: Adaptive EMA (KAMA-Based)](#feature-2-adaptive-ema-kama-based)
4. [Feature 3: Volatility-Adaptive Smoothing](#feature-3-volatility-adaptive-smoothing)
5. [Feature 4: Market Microstructure Noise Filter](#feature-4-market-microstructure-noise-filter)
6. [Feature 5: Regime Detection System](#feature-5-regime-detection-system)
7. [Feature 6: Dual-Mode Strategy Engine](#feature-6-dual-mode-strategy-engine)
8. [Feature 7: Lightweight State Management](#feature-7-lightweight-state-management)
9. [Feature 8: Async API Wrapper](#feature-8-async-api-wrapper)
10. [Feature 9: Backtesting Module](#feature-9-backtesting-module)
11. [Feature 10: Order Execution Engine](#feature-10-order-execution-engine)
12. [Feature 11: Risk Management System](#feature-11-risk-management-system)
13. [Feature 12: Frontend Dashboard (React + Next.js)](#feature-12-frontend-dashboard-react--nextjs)

---

## System Architecture

```
fluxhero/
├── backend/                      # Python backend (core logic)
│   ├── computation/              # Numba-optimized calculations
│   │   ├── indicators.py         # JIT-compiled indicators
│   │   ├── adaptive_ema.py       # KAMA & adaptive EMA logic
│   │   └── volatility.py         # ATR, volatility calculations
│   ├── strategy/                 # Trading strategy logic
│   │   ├── regime_detector.py   # Market regime classification
│   │   ├── dual_mode.py          # Trend + Mean reversion engine
│   │   └── signal_generator.py  # Buy/sell signal logic
│   ├── data/                     # Data pipeline
│   │   ├── fetcher.py            # API data retrieval (httpx/aiohttp)
│   │   └── cache.py              # Lightweight caching (Parquet)
│   ├── execution/                # Order management
│   │   ├── broker_interface.py  # Abstract broker interface
│   │   ├── order_manager.py     # Order lifecycle management
│   │   └── position_sizer.py    # Risk-based position sizing
│   ├── backtesting/              # Simulation engine
│   │   ├── engine.py             # Backtest orchestrator
│   │   ├── fills.py              # Next-bar fill logic
│   │   └── metrics.py            # Performance analytics
│   ├── risk/                     # Risk management
│   │   ├── position_limits.py   # Exposure limits
│   │   └── kill_switch.py       # Daily loss limits
│   ├── storage/                  # Lightweight persistence
│   │   ├── sqlite_store.py      # Trade logs, positions
│   │   └── parquet_store.py     # Historical data cache
│   ├── api/                      # REST API for frontend
│   │   └── server.py             # FastAPI/Flask server
│   └── config/                   # Configuration
│       └── settings.py           # System settings
├── frontend/                     # React + Next.js UI
│   ├── components/               # React components
│   ├── pages/                    # Next.js pages
│   ├── hooks/                    # Custom React hooks
│   └── lib/                      # Utilities
├── tests/                        # Test suites
└── requirements.txt              # Python dependencies
```

---

## Feature 1: JIT Computation Engine

### Purpose
Achieve near-C++ speeds for mathematical operations using Numba's Just-In-Time compilation without managing C++ build complexity.

### Requirements

#### 1.1 Core JIT Functions
- **R1.1.1**: All price-based loops (EMA, RSI, ATR calculations) must be decorated with `@njit`
- **R1.1.2**: Support vectorized operations for batch calculations (e.g., backtesting over 1000s of candles)
- **R1.1.3**: Type annotations must be explicit for Numba compatibility (`float64`, `int32` arrays)

#### 1.2 Performance Targets
- **R1.2.1**: Adaptive EMA calculation on 10,000 candles must complete in <100ms
- **R1.2.2**: Full indicator suite (EMA, ATR, RSI, KAMA) on 10,000 candles must complete in <500ms
- **R1.2.3**: Backtest simulation over 1 year of minute data (>100k candles) must complete in <10 seconds

#### 1.3 Optimization Strategy
- **R1.3.1**: Use `@njit(cache=True)` to cache compiled functions
- **R1.3.2**: Avoid Python objects inside JIT functions (use NumPy arrays only)
- **R1.3.3**: Implement parallel processing with `@njit(parallel=True)` for independent calculations

### Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| EMA (10k candles) | <100ms | `timeit` benchmark |
| Full indicator suite (10k candles) | <500ms | `timeit` benchmark |
| 1-year backtest (minute data) | <10s | End-to-end test |
| Speedup vs pure Python | >10x | Comparative benchmark |

### Dependencies
- `numba >= 0.58.0`
- `numpy >= 1.24.0`

---

## Feature 2: Adaptive EMA (KAMA-Based)

### Purpose
Create a self-adjusting EMA that responds faster in trends and slower in choppy markets using Kaufman's Adaptive Moving Average (KAMA) logic.

### Requirements

#### 2.1 Efficiency Ratio (ER) Calculation
- **R2.1.1**: Calculate ER as `ER = |Change| / Sum(|Price[i] - Price[i-1]|)`
  - `Change = Price[today] - Price[n_periods_ago]`
  - Denominator = sum of absolute price changes over n periods
- **R2.1.2**: ER must be bounded between 0 and 1
  - ER = 1 → Perfect trend (straight line)
  - ER = 0 → Pure noise (random walk)
- **R2.1.3**: Use configurable lookback period (default: 10 periods)

#### 2.2 Adaptive Smoothing Constant
- **R2.2.1**: Define fast and slow smoothing constants
  - `SC_fast = 2 / (2 + 1) = 0.6667` (2-period EMA)
  - `SC_slow = 2 / (30 + 1) = 0.0645` (30-period EMA)
- **R2.2.2**: Calculate adaptive smoothing constant (ASC):
  ```
  ASC = [ER × (SC_fast - SC_slow) + SC_slow]²
  ```
- **R2.2.3**: Use ASC in EMA calculation:
  ```
  KAMA[today] = KAMA[yesterday] + ASC × (Price[today] - KAMA[yesterday])
  ```

#### 2.3 Regime-Aware Adjustment
- **R2.3.1**: In trending markets (ER > 0.6), bias toward fast SC
- **R2.3.2**: In choppy markets (ER < 0.3), bias toward slow SC
- **R2.3.3**: Smooth transitions between regimes (no abrupt changes)

### Success Criteria

| Test Case | Expected Behavior | Pass Condition |
|-----------|------------------|----------------|
| Strong uptrend (10 consecutive +1% days) | ER > 0.7, ASC > 0.3 | KAMA follows price within 2% |
| Sideways chop (±0.5% oscillations) | ER < 0.3, ASC < 0.1 | KAMA stays flat, ignores noise |
| Trending → Choppy transition | ASC decreases smoothly over 5-10 bars | No sudden jumps in KAMA |
| Backtested Sharpe ratio improvement | KAMA-based strategy > Fixed EMA | +10% improvement |

### Mathematical Validation
- **Test 1**: Perfect trend (prices = [100, 101, 102, 103...]) → ER should be ~1.0
- **Test 2**: Pure noise (prices = [100, 99, 101, 100...]) → ER should be ~0.0
- **Test 3**: Verify ASC always between SC_slow and SC_fast

---

## Feature 3: Volatility-Adaptive Smoothing

### Purpose
Dynamically adjust indicator lookback periods based on ATR, allowing faster reactions in high-volatility environments.

### Requirements

#### 3.1 ATR-Based Period Adjustment
- **R3.1.1**: Calculate 14-period ATR as baseline volatility measure
- **R3.1.2**: Define volatility states:
  - Low volatility: ATR < 0.5 × ATR_MA(50)
  - Normal volatility: 0.5 × ATR_MA(50) ≤ ATR ≤ 1.5 × ATR_MA(50)
  - High volatility: ATR > 1.5 × ATR_MA(50)
- **R3.1.3**: Adjust indicator periods dynamically:
  ```python
  if ATR > 1.5 × ATR_MA:
      period = base_period × 0.7  # Shorten lookback by 30%
  elif ATR < 0.5 × ATR_MA:
      period = base_period × 1.3  # Lengthen lookback by 30%
  else:
      period = base_period
  ```

#### 3.2 Multi-Timeframe Volatility Check
- **R3.2.1**: Compare current 5-min ATR vs 1-hour ATR
- **R3.2.2**: If 5-min ATR > 2× hourly ATR → Flag "volatility spike"
- **R3.2.3**: During volatility spikes, widen stops by 1.5× and reduce position size by 30%

#### 3.3 Volatility-Alpha Linkage
- **R3.3.1**: High volatility (ATR rising) → Use high α (0.4-0.6) for fast reactions
- **R3.3.2**: Low volatility (ATR falling) → Use low α (0.1-0.2) to smooth noise
- **R3.3.3**: Use regression to find optimal α = f(ATR) relationship from historical data

### Success Criteria

| Scenario | System Response | Pass Condition |
|----------|----------------|----------------|
| ATR doubles in 10 minutes | Period shortens to 70% of base | Indicator captures move within 3 bars |
| ATR drops by 50% over 1 hour | Period lengthens to 130% of base | Reduced false signals by >20% |
| Flash crash (5% drop in 2 min) | Volatility spike flag + wide stops | No premature stop-outs |
| Backtest vs fixed periods | Adaptive system outperforms | +15% higher Sharpe ratio |

---

## Feature 4: Market Microstructure Noise Filter

### Purpose
Prevent trading on illiquid or manipulated price movements by validating spread-to-volatility ratios.

### Requirements

#### 4.1 Spread-to-Volatility Ratio
- **R4.1.1**: Calculate current bid-ask spread: `Spread = Ask - Bid`
- **R4.1.2**: Calculate recent 5-minute volatility: `Vol_5min = StdDev(Close, 20 bars on 5-min chart)`
- **R4.1.3**: Compute ratio: `SV_Ratio = Spread / Vol_5min`
- **R4.1.4**: Reject signals if `SV_Ratio > 0.05` (spread is >5% of recent volatility)

#### 4.2 Volume Validation
- **R4.2.1**: Calculate average volume over last 20 bars
- **R4.2.2**: Require current bar volume > 0.5 × Avg_Volume for signal validation
- **R4.2.3**: For breakout signals, require volume > 1.5 × Avg_Volume

#### 4.3 Time-of-Day Filter
- **R4.3.1**: Flag "illiquid hours":
  - Pre-market: Before 9:30 AM EST
  - Lunch: 12:00-1:00 PM EST
  - After-hours: After 4:00 PM EST
- **R4.3.2**: During illiquid hours, require 2× stricter SV_Ratio (<2.5%)
- **R4.3.3**: Disable new position entries 15 minutes before market close

### Success Criteria

| Test Case | Expected Outcome | Pass Condition |
|-----------|-----------------|----------------|
| Wide spread (0.10) + low vol (1.0) | Signal rejected (SV_Ratio = 10%) | No trade executed |
| Tight spread (0.02) + normal vol (2.0) | Signal allowed (SV_Ratio = 1%) | Trade executed |
| Low volume breakout | Signal rejected | False breakout avoided |
| Backtest with filter ON vs OFF | Fewer trades, higher win rate | +10% win rate improvement |

---

## Feature 5: Regime Detection System

### Purpose
Automatically classify market conditions as trending or mean-reverting to switch strategies dynamically.

### Requirements

#### 5.1 Trend Strength Measurement
- **R5.1.1**: Calculate ADX (Average Directional Index, 14-period)
  - ADX > 25 → Trending market
  - ADX < 20 → Ranging/choppy market
- **R5.1.2**: Calculate linear regression slope over 50 bars
  - R² > 0.7 → Strong linear trend
  - R² < 0.3 → No clear trend
- **R5.1.3**: Combine both metrics:
  ```python
  if ADX > 25 and R² > 0.6:
      regime = "STRONG_TREND"
  elif ADX < 20 and R² < 0.4:
      regime = "MEAN_REVERSION"
  else:
      regime = "NEUTRAL"
  ```

#### 5.2 Volatility Regime
- **R5.2.1**: Compare current ATR to 50-period ATR average
  - ATR > 1.5 × ATR_MA → "HIGH_VOL"
  - ATR < 0.7 × ATR_MA → "LOW_VOL"
- **R5.2.2**: Track regime persistence (how long in current state)
- **R5.2.3**: Require 3 consecutive bars to confirm regime change (avoid whipsaws)

#### 5.3 Correlation Check (Multi-Asset)
- **R5.3.1**: For portfolio strategies, check correlation between assets
- **R5.3.2**: If correlation > 0.8 → Markets moving together (risk-on/risk-off)
- **R5.3.3**: Adjust diversification: reduce position count if correlation spikes

### Success Criteria

| Regime | Detection Accuracy | Trading Outcome |
|--------|-------------------|-----------------|
| Strong uptrend (2020 tech rally) | >85% of days classified as TREND | Trend-following strategy active |
| Sideways (2015 oil crash recovery) | >80% of days classified as MEAN_REV | Mean reversion strategy active |
| Transition periods | <10% whipsaws (regime flip-flops) | Smooth strategy transitions |
| Backtest over 5 years | Regime-aware strategy beats buy-hold | +25% higher returns |

---

## Feature 6: Dual-Mode Strategy Engine

### Purpose
Implement both trend-following and mean-reversion logic, switching automatically based on detected regime.

### Requirements

#### 6.1 Trend-Following Mode
- **R6.1.1**: Entry signal: Price crosses above KAMA + (0.5 × ATR)
- **R6.1.2**: Exit signal: Price crosses below KAMA - (0.3 × ATR)
- **R6.1.3**: Trailing stop: ATR-based (2.5 × ATR from peak)
- **R6.1.4**: Position sizing: Risk 1% of capital per trade
- **R6.1.5**: Activate when regime = "STRONG_TREND"

#### 6.2 Mean-Reversion Mode
- **R6.2.1**: Entry signal: RSI < 30 AND price touches lower Bollinger Band
- **R6.2.2**: Exit signal: Price returns to 20-period SMA OR RSI > 70
- **R6.2.3**: Stop loss: Fixed 3% below entry
- **R6.2.4**: Position sizing: Risk 0.75% per trade (tighter risk for MR)
- **R6.2.5**: Activate when regime = "MEAN_REVERSION"

#### 6.3 Neutral/Transition Mode
- **R6.3.1**: When regime = "NEUTRAL", use blended approach:
  - 50% weight to trend signals
  - 50% weight to mean reversion signals
- **R6.3.2**: Reduce position sizes by 30% during neutral periods
- **R6.3.3**: Require both strategies to agree for entry (higher confidence threshold)

#### 6.4 Strategy Performance Tracking
- **R6.4.1**: Track win rate, Sharpe ratio, max drawdown for each mode separately
- **R6.4.2**: If one mode underperforms for 20+ trades, reduce its allocation
- **R6.4.3**: Monthly rebalance: adjust mode weights based on recent performance

### Success Criteria

| Test Scenario | Expected Strategy | Performance Target |
|---------------|------------------|-------------------|
| 2020-2021 bull run | Trend-following dominates (>70% of trades) | >30% annual return |
| 2015 choppy market | Mean reversion dominates (>60% of trades) | <15% max drawdown |
| 2022 bear market | Trend-following (short bias) | Positive returns in down year |
| 5-year backtest | Adaptive beats single-mode strategies | +40% higher risk-adjusted returns |

---

## Feature 7: Lightweight State Management

### Purpose
Store critical data (trades, positions, settings) without heavy database overhead using SQLite and Parquet.

### Requirements

#### 7.1 SQLite for Operational Data
- **R7.1.1**: Store in `data/system.db`:
  - `trades` table: entry/exit prices, timestamps, P&L
  - `positions` table: current open positions, unrealized P&L
  - `settings` table: system parameters, risk limits
- **R7.1.2**: Write trades immediately after execution (async write to avoid blocking)
- **R7.1.3**: Daily rollover: archive old trades to Parquet, keep only last 30 days in SQLite

#### 7.2 Parquet for Historical Data
- **R7.2.1**: Cache downloaded market data as `data/cache/{symbol}_{timeframe}.parquet`
- **R7.2.2**: On startup, check if cached data is <24 hours old; if yes, load from cache
- **R7.2.3**: Store only OHLCV + calculated indicators (EMA, ATR, RSI)
- **R7.2.4**: Compress with `snappy` codec for fast read/write

#### 7.3 No Historical Candle Storage
- **R7.3.1**: Fetch last 500 candles on startup via API
- **R7.3.2**: Maintain rolling 500-candle buffer in memory
- **R7.3.3**: Discard candles older than 500 bars (no long-term storage)

### Success Criteria

| Operation | Target Performance | Pass Condition |
|-----------|-------------------|----------------|
| Write trade to SQLite | <5ms | Non-blocking async write |
| Load 500 candles from Parquet | <50ms | Faster than API call |
| System startup (cold) | <3 seconds | Includes API fetch + cache load |
| System startup (warm cache) | <1 second | Pure cache load |
| Database size after 1 year | <100 MB | Efficient archival |

---

## Feature 8: Async API Wrapper

### Purpose
Non-blocking data fetching and live price streaming using `httpx` for REST and `websockets` for real-time feeds.

### Requirements

#### 8.1 REST API Client (httpx)
- **R8.1.1**: Async methods for:
  - `fetch_candles(symbol, timeframe, limit)` → Returns OHLCV data
  - `get_account_info()` → Returns balance, buying power
  - `place_order(symbol, qty, side, type)` → Submits order
- **R8.1.2**: Retry logic: 3 attempts with exponential backoff (1s, 2s, 4s)
- **R8.1.3**: Rate limiting: Respect broker's API limits (e.g., 200 requests/min for Alpaca)
- **R8.1.4**: Connection pooling: Reuse HTTP connections for efficiency

#### 8.2 WebSocket Live Feed
- **R8.2.1**: Subscribe to real-time price updates for active symbols
- **R8.2.2**: Reconnect automatically if connection drops (max 5 retries)
- **R8.2.3**: Process incoming ticks asynchronously (don't block main loop)
- **R8.2.4**: Heartbeat monitor: Alert if no data received for >60 seconds

#### 8.3 Data Pipeline
- **R8.3.1**: On startup:
  1. Fetch last 500 candles via REST (httpx)
  2. Open WebSocket connection
  3. Start processing live ticks
- **R8.3.2**: Every completed candle (e.g., hourly close), check for signals
- **R8.3.3**: Run signal checks in separate async task (don't block data feed)

### Success Criteria

| Test Case | Expected Behavior | Pass Condition |
|-----------|------------------|----------------|
| Fetch 500 candles | Completes without blocking | <2 seconds |
| WebSocket disconnect | Auto-reconnects within 5s | No data loss |
| API rate limit hit | Request queued, retried | No errors |
| Simultaneous API calls (10) | All complete via connection pool | <3 seconds total |

---

## Feature 9: Backtesting Module

### Purpose
Simulate trading strategies on historical data with realistic fill logic, slippage, and commission modeling.

### Requirements

#### 9.1 Next-Bar Fill Logic
- **R9.1.1**: Signal generation on bar N → Execution at bar N+1 open price
  - Example: Hourly signal at 10:00:00 → Fill at 10:01:00 (simulates 60s delay)
- **R9.1.2**: For minute data, use 1-bar delay
- **R9.1.3**: For daily data, use next day's open price
- **R9.1.4**: No "peeking" into future data (strict time-series split)

#### 9.2 Slippage & Commission Model
- **R9.2.1**: Fixed commission: $0.005 per share (Alpaca-like)
- **R9.2.2**: Slippage model:
  - Market orders: 0.01% penalty on entry, 0.01% loss on exit
  - Limit orders: Assume fill at limit price (optimistic)
- **R9.2.3**: Impact model for large orders:
  ```python
  if order_size > 0.1 × avg_volume:
      slippage += 0.05%  # Price impact penalty
  ```

#### 9.3 Metrics Calculation
- **R9.3.1**: Calculate after each backtest:
  - Total return, annualized return
  - Sharpe ratio (assume risk-free rate = 4%)
  - Max drawdown (peak-to-trough)
  - Win rate, avg win/loss ratio
  - Number of trades, holding period
- **R9.3.2**: Use `quantstats` library for detailed tearsheet
- **R9.3.3**: Export results to PDF report

#### 9.4 Walk-Forward Testing
- **R9.4.1**: Split data into 3-month training, 1-month testing windows
- **R9.4.2**: Roll forward: train on months 1-3, test on month 4; train on 2-4, test on 5, etc.
- **R9.4.3**: Aggregate results across all test periods
- **R9.4.4**: Strategy passes if >60% of test periods are profitable

### Success Criteria

| Metric | Minimum Target | Ideal Target |
|--------|---------------|--------------|
| Sharpe Ratio | >0.8 | >1.5 |
| Max Drawdown | <25% | <15% |
| Win Rate | >45% | >55% |
| Avg Win/Loss Ratio | >1.5 | >2.0 |
| Walk-forward pass rate | >60% periods profitable | >75% |

### Validation Tests
- **Test 1**: Buy-and-hold SPY should match actual SPY returns (±2%)
- **Test 2**: Perfect predictor (always right) should show realistic slippage impact
- **Test 3**: Random strategy should hover around breakeven (after costs)

---

## Feature 10: Order Execution Engine

### Purpose
Manage order lifecycle from signal generation to fill confirmation with broker-agnostic interface (future-proof for multiple brokers).

### Requirements

#### 10.1 Broker Interface (Abstract)
- **R10.1.1**: Define abstract methods:
  - `place_order(symbol, qty, side, order_type, limit_price=None)`
  - `cancel_order(order_id)`
  - `get_order_status(order_id)`
  - `get_positions()`
  - `get_account()`
- **R10.1.2**: Implement concrete classes later:
  - `AlpacaBroker` (for initial development)
  - `IBKRBroker` (future)
  - `PaperBroker` (for testing)

#### 10.2 Order Heartbeat Monitor
- **R10.2.1**: After placing order, poll status every 5 seconds
- **R10.2.2**: If unfilled after 60 seconds:
  - Cancel existing order
  - Recalculate mid-price
  - Resubmit at new mid-price (chase)
- **R10.2.3**: Max 3 chase attempts, then abandon

#### 10.3 Position Sizer
- **R10.3.1**: 1% Risk Rule:
  ```python
  shares = (account_balance × 0.01) / (entry_price - stop_loss_price)
  ```
- **R10.3.2**: Never exceed max position size:
  - 20% of account per position (diversification)
  - 50% total deployed capital (keep cash buffer)
- **R10.3.3**: Round down to nearest whole share (no fractional shares unless broker supports)

#### 10.4 Safety Kill-Switch
- **R10.4.1**: Track daily P&L since market open
- **R10.4.2**: If daily loss exceeds 3% of account:
  - Close all open positions immediately (market orders)
  - Disable new trade entries until next trading session
  - Send alert notification
- **R10.4.3**: Manual override: require explicit user action to re-enable

### Success Criteria

| Feature | Test Case | Pass Condition |
|---------|-----------|----------------|
| Order placement | Submit 10 orders in 1 minute | All accepted, no errors |
| Heartbeat monitor | Order unfilled for 65s | Auto-canceled and rechased |
| Position sizing | $100k account, $50 entry, $48 stop | Calculates 500 shares (1% risk = $1k) |
| Kill-switch | Lose 3.1% in one day | All positions closed, trading disabled |

---

## Feature 11: Risk Management System

### Purpose
Protect capital through position sizing, exposure limits, drawdown controls, and correlation monitoring.

### Requirements

#### 11.1 Position-Level Risk
- **R11.1.1**: Max risk per trade: 1% of account (trend), 0.75% (mean reversion)
- **R11.1.2**: Max position size: 20% of account value
- **R11.1.3**: Stop loss mandatory on all positions (no exceptions)
- **R11.1.4**: ATR-based stops:
  - Trend trades: 2.5 × ATR
  - Mean reversion trades: 3% fixed stop

#### 11.2 Portfolio-Level Risk
- **R11.2.1**: Max total exposure: 50% of account (other 50% in cash)
- **R11.2.2**: Max open positions: 5 simultaneously
- **R11.2.3**: Correlation check:
  - Before opening new position, check correlation with existing positions
  - If correlation > 0.7 with any open position, reduce new position size by 50%

#### 11.3 Drawdown Circuit Breakers
- **R11.3.1**: Track drawdown from equity peak
- **R11.3.2**: At 15% drawdown:
  - Reduce all new position sizes by 50%
  - Tighten stops to 2.0 × ATR (from 2.5 × ATR)
  - Send warning alert
- **R11.3.3**: At 20% drawdown:
  - Close all positions
  - Disable trading
  - Require manual review before resuming

#### 11.4 Risk Monitoring Dashboard
- **R11.4.1**: Real-time display:
  - Current drawdown % from peak
  - Total exposure %
  - Risk per position
  - Correlation matrix (if multiple positions)
- **R11.4.2**: Daily risk report:
  - Total risk deployed (sum of all stop distances)
  - Largest position size
  - Worst-case scenario loss (if all stops hit)

### Success Criteria

| Scenario | System Response | Pass Condition |
|----------|----------------|----------------|
| 5 consecutive losing trades | Drawdown <6% (5 × 1% + slippage) | No circuit breaker triggered |
| 15% drawdown | Position sizes cut 50%, stops tightened | Prevents further decline |
| Attempt to open correlated position | Size reduced or rejected | Maintains diversification |
| Backtest over worst market period | Max drawdown <25% | System survives 2008/2020 crashes |

---

## Feature 12: Frontend Dashboard (React + Next.js)

### Purpose
Provide real-time visibility into system state, positions, and performance via a modern web interface.

### Requirements

#### 12.1 Architecture
- **R12.1.1**: Separate frontend repo/folder from Python backend
- **R12.1.2**: Communication via REST API (FastAPI backend)
- **R12.1.3**: WebSocket connection for live price updates
- **R12.1.4**: Deployed independently (frontend: Vercel, backend: local/VPS)

#### 12.2 Page Structure

##### Tab A: Live Trading
- **R12.2.1**: Open Positions Table
  - Columns: Symbol, Entry Price, Current Price, P&L ($), P&L (%), Time Held
  - Color coding: Green (profit), Red (loss)
- **R12.2.2**: System Heartbeat Indicator
  - Last data received timestamp
  - Status: 🟢 Active / 🟡 Delayed (>30s) / 🔴 Offline (>60s)
- **R12.2.3**: Quick Stats
  - Daily P&L
  - Current drawdown %
  - Total exposure %

##### Tab B: Analytics
- **R12.2.4**: Candlestick Chart (TradingView-like)
  - Use `react-stockcharts` or `lightweight-charts` library
  - Overlay: KAMA line, ATR bands
  - Annotations: Buy/sell signals as arrows
- **R12.2.5**: Indicator Panel
  - Real-time values: ATR, RSI, ADX, Regime state
  - Historical chart of each indicator
- **R12.2.6**: Performance Metrics
  - Total return (%), Sharpe ratio, Win rate
  - Max drawdown, Avg holding period

##### Tab C: Trade History
- **R12.2.7**: Trade Log Table
  - Columns: Date, Symbol, Side, Entry, Exit, P&L, Holding Time
  - Pagination (20 trades per page)
  - Export to CSV button

##### Tab D: Backtesting
- **R12.2.8**: Backtest Configuration Form
  - Start/end date pickers
  - Symbol selector
  - Strategy parameters (dropdowns/sliders)
- **R12.2.9**: Run Backtest Button
  - Shows loading spinner
  - Displays results in modal/new page
  - Includes tearsheet (via `quantstats` HTML export)

#### 12.3 Signal Explainer
- **R12.3.1**: For each trade, log reason:
  ```
  "BUY SPY @ $420.50
  Reason: Volatility (ATR=3.2, High) + KAMA crossover (Price > KAMA+0.5×ATR)
  Regime: STRONG_TREND (ADX=32, R²=0.81)
  Risk: $1,000 (1% account), Stop: $415.00"
  ```
- **R12.3.2**: Display in tooltip on hover over trade in table
- **R12.3.3**: Archive explanations for post-trade analysis

#### 12.4 UI/UX Requirements
- **R12.4.1**: Responsive design (works on desktop + tablet)
- **R12.4.2**: Dark mode support (toggle in settings)
- **R12.4.3**: Auto-refresh live data every 5 seconds
- **R12.4.4**: Loading states for async operations
- **R12.4.5**: Error boundaries for graceful error handling

### Success Criteria

| Feature | Test | Pass Condition |
|---------|------|----------------|
| Live updates | Open position P&L updates within 5s of price change | Visual confirmation |
| Chart rendering | Load 500-candle chart with indicators | <2s load time |
| Backtest run | Execute 1-year backtest and display results | <15s end-to-end |
| Mobile responsiveness | View on iPad | All tables readable, no horizontal scroll |

---

## Implementation Priority

### Phase 1: Core Engine (Weeks 1-3)
1. ✅ Set up project structure (backend + frontend folders)
2. ✅ Implement JIT computation engine (Numba)
3. ✅ Build adaptive EMA (KAMA logic)
4. ✅ Add volatility-adaptive smoothing
5. ✅ Create regime detection system

### Phase 2: Data & Backtesting (Weeks 4-6)
6. ✅ Async API wrapper (httpx + websockets)
7. ✅ Lightweight storage (SQLite + Parquet)
8. ✅ Backtesting engine (next-bar fills, slippage)
9. ✅ Metrics calculation (quantstats integration)

### Phase 3: Trading Logic (Weeks 7-8)
10. ✅ Dual-mode strategy engine
11. ✅ Market microstructure filter
12. ✅ Position sizer
13. ✅ Risk management system

### Phase 4: Execution & Monitoring (Weeks 9-10)
14. ✅ Order execution engine (paper trading first)
15. ✅ Kill-switch & circuit breakers
16. ✅ Signal explainer logging

### Phase 5: Frontend (Weeks 11-12)
17. ✅ Next.js setup + API integration
18. ✅ Live trading tab
19. ✅ Analytics tab (charts)
20. ✅ Backtest interface

---

## Retail-Specific Considerations

### The "Price Gap" Handling
- **Issue**: Stocks gap 5%+ on earnings, triggering false signals
- **Solution**:
  - Before entry, check: `|Open - Prev_Close| / Prev_Close < 0.02` (2% max gap)
  - If gap > 2%, skip trade even if signal fires
  - Add to Feature 4 (Noise Filter)

### The "Hourly Close" Optimization
- **Issue**: Most retail traders trade on candle close, causing slippage
- **Solution**:
  - Check signal at minute 59 of the hour (e.g., 10:59:00)
  - If signal valid, prepare order to submit at 11:00:00 exactly
  - Gives 60-second headstart vs. traders waiting for 11:00:01 close
  - Add to Feature 10 (Execution)

### Daily Reboot Task
- **Issue**: Memory leaks, stale WebSocket connections
- **Solution**:
  - Schedule daily restart at 9:00 AM EST (before market open)
  - Use `cron` (Linux) or `Task Scheduler` (Windows)
  - On startup: fetch last 500 candles, reconnect WebSocket, resume trading
  - Add to system operations documentation

---

## Maintenance & Operational Costs

### Data Costs
| Provider | Tier | Cost | Notes |
|----------|------|------|-------|
| Alpaca | Free | $0/mo | Real-time data for paper trading |
| Polygon.io | Starter | $29/mo | Real-time stock data (for live trading) |
| Alternative | Free tier | $0/mo | Delayed data acceptable for swing trading |

### Hosting Costs
| Option | Cost | Specs | Use Case |
|--------|------|-------|----------|
| Home PC | $0/mo | DIY | Development, paper trading |
| DigitalOcean VPS | $10/mo | 2GB RAM, 1 vCPU | Small live trading |
| AWS Lightsail | $12/mo | 2GB RAM, 1 vCPU | Alternative to DO |
| Vercel (Frontend) | $0/mo | Free tier | Frontend hosting |

### Code Maintenance
- **Python dependencies**: `pip freeze > requirements.txt` (version lock)
- **Update schedule**: Quarterly dependency updates (not automatic)
- **Backup**: Daily export of trades table to CSV (stored in Git)
- **Monitoring**: Email alerts on errors (via `smtplib`)

**Total estimated cost**: $0-40/month depending on data provider and hosting choice

---

## Next Steps

1. ✅ Review and approve this requirements document
2. ✅ Set up Python environment (`python -m venv venv`)
3. ✅ Install core dependencies (`numba`, `numpy`, `pandas`, `httpx`)
4. ✅ Begin implementation of Feature 1 (JIT Computation Engine)
5. ✅ Build unit tests alongside each feature (TDD approach)

---

*Document Version: 1.0*
*Last Updated: 2026-01-19*
*Status: DRAFT - Awaiting Approval*
