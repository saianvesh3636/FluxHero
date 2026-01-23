# FluxHero Trading System Assumptions

This document catalogs all assumptions made in the FluxHero backtesting and trading system. Each assumption includes the rationale, alternatives considered, and implications for interpretation of results.

---

## Table of Contents

1. [Return Calculations](#return-calculations)
2. [Commission Model](#commission-model)
3. [Slippage Model](#slippage-model)
4. [Order Fill Assumptions](#order-fill-assumptions)
5. [Position Sizing and Risk Model](#position-sizing-and-risk-model)
6. [Risk-Free Rate](#risk-free-rate)
7. [Data Quality](#data-quality)
8. [Market Assumptions](#market-assumptions)

---

## Return Calculations

### Simple Returns (Current Implementation)

**Location:** `backend/backtesting/metrics.py` (lines 26-61)

**Formula:**
```
return[t] = (equity[t] - equity[t-1]) / equity[t-1]
```

**Rationale:**
- Simple returns are intuitive and directly represent percentage change
- Easy to interpret for non-technical stakeholders
- Standard for reporting portfolio performance

**Alternatives Considered:**
- **Log returns:** `log(equity[t] / equity[t-1])`
  - More normally distributed (better for statistical analysis)
  - Additive across time (convenient for multi-period analysis)
  - Better for high-frequency or long-horizon analysis
  - Trade-off: Less intuitive interpretation

**Implications:**
- Simple returns slightly overstate compounded gains over long periods
- For typical backtest horizons (1-5 years), the difference is minimal
- Statistical tests (Sharpe ratio normality assumptions) may be slightly less accurate

**Future Enhancement:**
- See enhancement_tasks.md Phase 20 for planned log returns support

---

## Commission Model

### Per-Share Commission: $0.005/share

**Location:** `backend/backtesting/engine.py` (line 76)

**Configuration:**
```python
commission_per_share: float = 0.005  # $0.005 per share
```

**Rationale:**
- Modeled after Alpaca Markets commission structure
- Represents typical discount broker rates for active traders
- Conservative estimate that accounts for regulatory fees

**Breakdown of Real-World Costs:**
| Fee Type | Typical Cost | Our Model |
|----------|-------------|-----------|
| Broker Commission | $0.00 (most brokers) | Included |
| SEC Fee | ~$0.0000278/$ sold | Approximated |
| FINRA TAF | $0.000145/share (max $7.27) | Approximated |
| Exchange Fees | Varies | Approximated |

**Alternatives Considered:**
- **Zero commission:** Too optimistic; ignores regulatory fees
- **Per-trade flat fee:** Less accurate for varying position sizes
- **Percentage-based:** Less common for equity trading

**Implications:**
- 100-share trade = $0.50 round-trip commission
- 1000-share trade = $5.00 round-trip commission
- Commission drag is minimal for position sizes > $10,000
- May underestimate costs for very small accounts

---

## Slippage Model

### Base Slippage: 0.01% (1 basis point)

**Location:** `backend/backtesting/engine.py` (lines 77-79)

**Configuration:**
```python
slippage_pct: float = 0.0001       # 0.01% base slippage
impact_penalty_pct: float = 0.0005 # 0.05% additional for large orders
impact_threshold: float = 0.1      # 10% of average volume
```

**Implementation:**
- BUY orders: `fill_price = open_price * (1 + slippage)`
- SELL orders: `fill_price = open_price * (1 - slippage)`
- Large orders (>10% avg volume): additional 0.05% penalty

**Rationale:**
- 0.01% represents typical bid-ask spread for liquid securities (SPY, QQQ)
- Market impact model captures large order effects
- Conservative but realistic for retail-sized orders

**Empirical Basis:**
| Security Type | Typical Spread | Our Model |
|--------------|----------------|-----------|
| SPY, QQQ | 0.01-0.02% | 0.01% |
| Large-cap stocks | 0.02-0.05% | 0.01% (optimistic) |
| Mid-cap stocks | 0.05-0.15% | 0.01% (optimistic) |
| Small-cap stocks | 0.10-0.50% | 0.01% (very optimistic) |

**Alternatives Considered:**
- **Zero slippage:** Unrealistic; overstates performance
- **Fixed dollar amount:** Less accurate across price ranges
- **Volume-weighted:** More accurate but complex

**Implications:**
- Results may overstate performance for less liquid securities
- Market impact penalty helps account for larger positions
- For SPY/QQQ backtests, this is a reasonable estimate
- For individual stocks, consider increasing slippage

---

## Order Fill Assumptions

### Next-Bar Open Fill

**Location:** `backend/backtesting/fills.py` (lines 20-66)

**Rule:** Signal generated on bar N fills at bar N+1 open price

**Configuration:**
```python
FILL_DELAY_BARS = 1  # Signal on bar N -> fill at bar N+1 open
```

**Rationale:**
- Prevents look-ahead bias (cannot trade on current bar's close)
- Represents realistic order execution latency
- For daily bars: signal EOD, fill next morning open
- For minute bars: signal at close, fill 1 minute later

**Execution Timeline:**
```
Bar N Close: Strategy evaluates, generates signal
Bar N+1 Open: Order executes at open price (+ slippage)
```

**Stop Loss and Take Profit:**
- Stop losses can execute intrabar at the stop price
- Take profits can execute intrabar at the target price
- Both use conservative fill assumptions

**Alternatives Considered:**
- **Same-bar close fill:** Creates look-ahead bias
- **VWAP fill:** More realistic but requires intrabar data
- **Limit orders:** Would require order book simulation

**Implications:**
- Gap risk is captured (overnight gaps for daily data)
- Fast-moving markets may see worse fills than modeled
- For minute data, 1-minute delay is conservative

---

## Position Sizing and Risk Model

### Risk-Based Position Sizing

**Location:** `backend/risk/position_limits.py` (lines 104-149)

**Formula:**
```
shares = (account_balance * risk_pct) / |entry_price - stop_loss|
```

**Risk Parameters by Strategy:**

| Strategy Type | Risk Per Trade | Stop Loss | Max Position | Max Deployment |
|--------------|----------------|-----------|--------------|----------------|
| Trend Following | 1.0% | 2.5 * ATR | 20% | 50% |
| Mean Reversion | 0.75% | 3.0% fixed | 20% | 50% |

**Position Limits:**
```python
max_single_position_pct: 0.20   # 20% max in single position
max_total_deployment_pct: 0.50  # 50% max deployed at once
max_open_positions: 5           # Maximum concurrent positions
```

**Rationale:**
- 1% risk per trade is industry standard for systematic trading
- Lower risk (0.75%) for mean reversion due to higher frequency
- 20% position limit prevents overconcentration
- 50% deployment limit keeps dry powder for opportunities

**Kelly Criterion Comparison:**
- Full Kelly often suggests 10-25% risk (too aggressive)
- Our 1% is approximately 1/10th Kelly (conservative)
- Reduces drawdowns at cost of lower expected returns

**Alternatives Considered:**
- **Fixed dollar amount:** Doesn't scale with account size
- **Fixed share count:** Doesn't account for price differences
- **Volatility targeting:** More complex, considered for v2

**Implications:**
- Maximum single-trade loss is 1% of account
- Worst-case 5-position loss is 5% of account
- Compound growth may be slower than aggressive sizing
- Drawdowns should be manageable (<20% typical)

---

## Risk-Free Rate

### Annual Risk-Free Rate: 4.0%

**Location:** `backend/backtesting/engine.py` (line 80)

**Configuration:**
```python
risk_free_rate: float = 0.04  # 4% annual risk-free rate
```

**Usage:**
- Sharpe ratio calculation: `(returns - risk_free) / volatility`
- Risk-adjusted return metrics

**Rationale:**
- Approximates current US Treasury rates (2024-2025)
- Conservative estimate for opportunity cost of capital
- Appropriate for USD-denominated strategies

**Historical Context:**
| Period | 10Y Treasury | Our Model |
|--------|-------------|-----------|
| 2020-2021 | 0.5-1.5% | 4.0% (conservative) |
| 2022-2023 | 2.5-4.5% | 4.0% (appropriate) |
| 2024-2025 | 3.5-4.5% | 4.0% (appropriate) |

**Alternatives Considered:**
- **0%:** Common but overstates risk-adjusted returns
- **Dynamic rate:** More accurate but adds complexity
- **SOFR-based:** Better for short-term comparisons

**Implications:**
- Sharpe ratios will appear lower than zero-rate calculations
- Strategies must earn >4% to show positive excess returns
- More conservative than many backtest frameworks

---

## Data Quality

### Data Validation Assumptions

**Location:** `backend/data/yahoo_provider.py` and `backend/backtesting/engine.py`

**Validation Rules:**
- No NaN values in OHLCV data
- No negative prices
- No zero volume bars (unless market closed)
- High >= Low for all bars
- No gaps > 5 trading days (warns, doesn't fail)

**Survivorship Bias:**
- Yahoo Finance data includes delisted securities
- Historical constituents not tracked (potential bias)
- For index ETFs (SPY, QQQ), survivorship bias is minimal

**Dividend/Split Handling:**
- Uses adjusted close prices from Yahoo Finance
- Assumes dividends reinvested (total return)
- Splits are pre-adjusted in historical data

**Implications:**
- Individual stock backtests may have survivorship bias
- ETF backtests are more reliable
- Dividend reinvestment assumption may not match reality

---

## Market Assumptions

### Trading Hours and Liquidity

**Assumptions:**
- All orders execute during regular market hours
- Sufficient liquidity exists for all orders
- No partial fills (all-or-nothing execution)
- No trading halts or circuit breakers modeled

**Market Regime:**
- Backtests assume markets are open and functioning normally
- Flash crashes and extreme events not specifically modeled
- Regime detection uses historical data (lagging indicator)

**Correlation Assumptions:**
```python
correlation_threshold: 0.7        # Assets >0.7 correlated = same risk
correlation_size_reduction: 0.5   # 50% size reduction for correlated assets
```

**Implications:**
- Results may not hold during extreme market stress
- Black swan events could cause larger losses than modeled
- Regime changes may take 3+ bars to detect (intentional lag)

---

## Summary Table

| Category | Assumption | Value | Requirement Ref |
|----------|------------|-------|-----------------|
| Returns | Simple percentage returns | `(P1-P0)/P0` | - |
| Commission | Per-share cost | $0.005/share | R9.2.1 |
| Slippage (base) | Percentage of price | 0.01% | R9.2.2 |
| Slippage (impact) | Large order penalty | 0.05% if >10% vol | R9.2.3 |
| Fill timing | Bars after signal | 1 bar (next open) | R9.1.1 |
| Risk (trend) | Per-trade risk | 1.0% | R11.1.1 |
| Risk (mean-rev) | Per-trade risk | 0.75% | R11.1.1 |
| Stop (trend) | ATR multiple | 2.5x ATR | R11.1.4 |
| Stop (mean-rev) | Fixed percentage | 3.0% | R11.1.4 |
| Max position | Single position limit | 20% | R11.1.2 |
| Max deployment | Total deployed limit | 50% | R11.1.2 |
| Max positions | Concurrent positions | 5 | R11.2.2 |
| Risk-free rate | Annual rate | 4.0% | R9.3.1 |

---

## Updating Assumptions

When modifying any assumption in the codebase:
1. Update the relevant section in this document
2. Add an `# ASSUMPTION:` comment at the code location
3. Consider the impact on historical backtest results
4. Document the change in `.context/history.md`

---

*Last updated: 2026-01-23*
