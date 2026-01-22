# FluxHero Risk Management System

## Overview

The FluxHero Risk Management System implements comprehensive capital protection through position sizing, exposure limits, drawdown controls, and correlation monitoring. This document describes all risk rules and circuit breaker behaviors implemented in the system.

**Location**: `fluxhero/backend/risk/`

**Key Modules**:
- `position_limits.py` - Position-level and portfolio-level risk controls
- `kill_switch.py` - Drawdown circuit breakers and kill switch

---

## Table of Contents

1. [Position-Level Risk Controls](#position-level-risk-controls)
2. [Portfolio-Level Risk Controls](#portfolio-level-risk-controls)
3. [Drawdown Circuit Breakers](#drawdown-circuit-breakers)
4. [Risk Monitoring](#risk-monitoring)
5. [Configuration](#configuration)
6. [Examples](#examples)
7. [Testing and Validation](#testing-and-validation)

---

## Position-Level Risk Controls

### R11.1.1: Maximum Risk Per Trade

**Rule**: Risk a maximum percentage of account value per trade, based on strategy type.

**Limits**:
- **Trend-Following Trades**: 1.0% of account value
- **Mean-Reversion Trades**: 0.75% of account value
- **Neutral Mode**: 0.75% of account value (conservative)

**Formula**:
```
shares = (account_balance × risk_pct) / |entry_price - stop_loss|
```

**Example**:
- Account Balance: $100,000
- Entry Price: $50.00
- Stop Loss: $48.00
- Strategy: Trend-Following (1% risk)
- Risk Amount: $100,000 × 0.01 = $1,000
- Price Risk: $50.00 - $48.00 = $2.00
- **Position Size: $1,000 / $2.00 = 500 shares**

**Implementation**: `calculate_position_size_from_risk()` in `position_limits.py:91-144`

**Validation**: `validate_position_level_risk()` checks that calculated risk does not exceed limits

**Rejection**: Trade is rejected with `RiskCheckResult.REJECTED_EXCESSIVE_RISK` if risk exceeds limits

---

### R11.1.2: Maximum Position Size

**Rule**: No single position can exceed 20% of total account value.

**Limit**: 20% of account balance

**Formula**:
```
max_position_value = account_balance × 0.20
position_value = shares × entry_price
```

**Example**:
- Account Balance: $100,000
- Maximum Position Value: $20,000
- Entry Price: $100.00
- **Maximum Shares: 200 shares**

**Implementation**: `validate_position_level_risk()` in `position_limits.py:147-210`

**Rejection**: Trade is rejected with `RiskCheckResult.REJECTED_POSITION_TOO_LARGE` if position size exceeds 20%

**Rationale**: Prevents over-concentration in a single security. Even if risk per trade is small (e.g., tight stop), total exposure must be limited.

---

### R11.1.3: Mandatory Stop Loss

**Rule**: Every position MUST have a stop loss. No exceptions.

**Validation**: Stop loss parameter cannot be `None` when opening a position.

**Implementation**: `validate_position_level_risk()` checks for `stop_loss is None`

**Rejection**: Trade is rejected with `RiskCheckResult.REJECTED_NO_STOP` if no stop loss is provided

**Rationale**: Protects against unlimited loss scenarios. All positions must have predefined exit points.

---

### R11.1.4: ATR-Based Stop Loss Calculation

**Rule**: Stop losses are calculated based on market volatility (ATR) or fixed percentage.

**Formulas**:

**Trend-Following Trades** (2.5× ATR):
```
stop_distance = ATR × 2.5
stop_loss = entry_price - (stop_distance × side)
  where side = 1 for long, -1 for short
```

**Mean-Reversion Trades** (3% fixed):
```
stop_distance = entry_price × 0.03
stop_loss = entry_price - (stop_distance × side)
```

**Examples**:

**Trend Trade (Long)**:
- Entry: $100.00
- ATR: $2.00
- Stop Distance: $2.00 × 2.5 = $5.00
- **Stop Loss: $95.00**

**Mean-Reversion Trade (Long)**:
- Entry: $100.00
- Stop Distance: $100.00 × 0.03 = $3.00
- **Stop Loss: $97.00**

**Implementation**: `calculate_atr_stop_loss()` in `position_limits.py:213-253`

**Rationale**:
- **Trend trades**: Wider stops (2.5× ATR) allow trends to breathe and avoid premature exits
- **Mean-rev trades**: Tighter fixed stops (3%) since mean-reversion is shorter-term

---

## Portfolio-Level Risk Controls

### R11.2.1: Maximum Total Exposure

**Rule**: Total deployed capital cannot exceed 50% of account value. The other 50% must remain in cash.

**Limit**: 50% of account balance

**Formula**:
```
current_exposure = sum(position.market_value for all open positions)
new_exposure = current_exposure + new_position_value
max_exposure = account_balance × 0.50
```

**Example**:
- Account Balance: $100,000
- Maximum Total Exposure: $50,000
- Current Open Positions: $35,000
- **Available for New Positions: $15,000**

**Implementation**: `validate_portfolio_level_risk()` in `position_limits.py:260-305`

**Rejection**: Trade is rejected with `RiskCheckResult.REJECTED_TOTAL_EXPOSURE` if total exposure would exceed 50%

**Rationale**: Maintains cash reserves for:
- Managing margin calls
- Taking advantage of new opportunities
- Surviving drawdown periods without forced liquidation

---

### R11.2.2: Maximum Open Positions

**Rule**: No more than 5 positions can be open simultaneously.

**Limit**: 5 positions

**Implementation**: `validate_portfolio_level_risk()` counts open positions

**Rejection**: Trade is rejected with `RiskCheckResult.REJECTED_MAX_POSITIONS` if already at 5 positions

**Rationale**:
- Prevents over-diversification ("diworsification")
- Ensures each position is meaningful (minimum ~10% allocation if fully deployed)
- Reduces monitoring complexity
- Maintains focus on highest-conviction opportunities

---

### R11.2.3: Correlation Monitoring

**Rule**: Check correlation between new position and existing positions. Reduce position size if highly correlated.

**Threshold**: Correlation > 0.7 (absolute value)

**Action**: Reduce new position size by 50% if correlation exceeds threshold

**Formula**:
```
correlation = cov(prices1, prices2) / (std(prices1) × std(prices2))
if |correlation| > 0.7:
    adjusted_shares = shares × 0.50
```

**Example**:
- Existing Position: SPY (S&P 500)
- New Position: QQQ (Nasdaq 100)
- Calculated Correlation: 0.85
- Original Position Size: 1000 shares
- **Adjusted Position Size: 500 shares (50% reduction)**

**Implementation**:
- `calculate_correlation()` in `position_limits.py:308-361` (Numba-optimized Pearson correlation)
- `check_correlation_with_existing_positions()` in `position_limits.py:364-426`

**Rationale**:
- Prevents false diversification (holding two highly correlated assets)
- Reduces portfolio risk during market-wide moves
- Correlation > 0.7 indicates assets move together ~70% of the time

**Note**: Correlation is calculated using recent price history (typically last 50-100 bars). Minimum 2 bars required.

---

## Drawdown Circuit Breakers

### Overview

Circuit breakers automatically reduce risk or halt trading when account drawdown reaches critical levels. This prevents catastrophic losses during adverse market conditions.

**Drawdown Formula**:
```
drawdown_pct = (equity_peak - current_equity) / equity_peak
```

**Drawdown Levels**:
- **NORMAL**: < 15% drawdown (normal operations)
- **WARNING**: 15-20% drawdown (reduced risk mode)
- **CRITICAL**: ≥ 20% drawdown (trading disabled)

---

### R11.3.1: Drawdown Tracking

**Rule**: Continuously track account equity peak and current drawdown from peak.

**Tracking**:
- **Equity Peak**: Highest account equity ever reached
- **Current Equity**: Current account balance + unrealized P&L
- **Drawdown**: Equity Peak - Current Equity
- **Drawdown %**: Drawdown / Equity Peak

**Implementation**: `EquityTracker` class in `kill_switch.py:131-216`

**Methods**:
- `update_equity(new_equity)`: Updates current equity and recalculates peak if new high
- `calculate_drawdown_pct()`: Returns current drawdown percentage
- `reset_peak(new_peak)`: Manually reset peak (use with extreme caution)

**Example**:
```python
tracker = EquityTracker(initial_equity=100000)
tracker.update_equity(105000)  # New peak: $105,000
tracker.update_equity(95000)   # Drawdown: $10,000 (9.52%)
```

---

### R11.3.2: 15% Drawdown - Warning Level

**Rule**: When drawdown reaches 15%, automatically reduce risk exposure.

**Trigger**: `drawdown_pct >= 0.15`

**Actions**:
1. **Reduce New Position Sizes by 50%**
   - All new trades are sized at 50% of normal calculation
   - Existing positions remain unchanged

2. **Tighten Stop Losses**
   - Normal stops: 2.5× ATR
   - Warning stops: 2.0× ATR
   - Applies to new positions only

3. **Send Warning Alert**
   - Alert message: "WARNING: 15% drawdown reached. Position sizes reduced 50%, stops tightened to 2.0× ATR."
   - Alert is logged with timestamp

**Trading Status**: `TradingStatus.REDUCED`

**Example**:
- Normal Position Size: 1000 shares
- Warning Position Size: 500 shares
- Normal Stop: Entry - (2.5 × ATR)
- Warning Stop: Entry - (2.0 × ATR)

**Implementation**:
- `DrawdownCircuitBreaker.update_trading_status()` in `kill_switch.py:268-331`
- `get_position_size_multiplier()` returns 0.5 in `kill_switch.py:333-354`
- `get_stop_loss_multiplier()` returns 2.0 in `kill_switch.py:356-375`

**Recovery**: When drawdown falls below 15%, trading automatically returns to normal mode with alert.

**Rationale**:
- Reduces risk before reaching critical levels
- Tighter stops limit further losses
- Preserves capital during adverse periods
- Automatic recovery when conditions improve

---

### R11.3.3: 20% Drawdown - Critical Level (Kill Switch)

**Rule**: When drawdown reaches 20%, immediately halt all trading activity.

**Trigger**: `drawdown_pct >= 0.20`

**Actions**:
1. **Close All Open Positions**
   - Market orders to exit all positions immediately
   - Executed at next available price

2. **Disable All Trading**
   - No new positions can be opened
   - Trading status set to `DISABLED`

3. **Require Manual Review**
   - `manual_review_required` flag set to `True`
   - Trading cannot resume until manual acknowledgment

4. **Send Critical Alert**
   - Alert message: "CRITICAL: 20% drawdown reached. All trading disabled. Manual review required."
   - Alert is logged with timestamp

**Trading Status**: `TradingStatus.DISABLED`

**Manual Acknowledgment**:
```python
circuit_breaker.acknowledge_manual_review()
# Only call this after:
# 1. Reviewing what went wrong
# 2. Adjusting strategy parameters
# 3. Confirming market conditions improved
```

**Implementation**:
- `DrawdownCircuitBreaker.update_trading_status()` in `kill_switch.py:268-331`
- `should_close_all_positions()` returns `True` in `kill_switch.py:401-411`
- `can_open_new_position()` returns `(False, reason)` in `kill_switch.py:377-399`
- `acknowledge_manual_review()` clears flag in `kill_switch.py:413-421`

**Example Scenario**:
1. Account starts at $100,000 (peak)
2. Series of losses reduces account to $80,000
3. Drawdown = 20% → Kill switch activates
4. All positions immediately closed
5. Trading disabled until manual review
6. Operator reviews logs, adjusts parameters, acknowledges review
7. Trading resumes with new peak set

**Rationale**:
- **20% is a significant loss** - requires human intervention
- **Prevents catastrophic losses** - automatic stop prevents account blowup
- **Forces reflection** - manual review ensures problems are addressed, not ignored
- **Common in institutional trading** - similar to portfolio manager stop-loss limits

**Warning**: Do NOT bypass this mechanism. It exists to protect capital. If the kill switch triggers, something went wrong and requires investigation.

---

## Risk Monitoring

### R11.4.1: Real-Time Risk Display

**Purpose**: Provide continuous visibility into risk metrics during live trading.

**Metrics Displayed**:

1. **Current Drawdown**
   - Drawdown amount (dollars)
   - Drawdown percentage
   - Drawdown level (NORMAL/WARNING/CRITICAL)

2. **Total Exposure**
   - Sum of all position market values
   - Exposure as percentage of account
   - Number of open positions

3. **Risk Per Position**
   - Position symbol
   - Market value
   - Stop loss price
   - Risk amount (if stop hit)
   - Risk as % of account

4. **Correlation Matrix** (if multiple positions)
   - Pairwise correlation between all positions
   - Highlights correlations > 0.7 (high risk)

**Implementation**: `calculate_risk_metrics()` in `kill_switch.py:440-513`

**Update Frequency**: Real-time (updates every time equity changes)

**Display**: Rendered in frontend dashboard (Live Trading tab)

---

### R11.4.2: Daily Risk Report

**Purpose**: Generate comprehensive daily summary of risk exposure.

**Report Contents**:

1. **Account Status**
   - Current balance
   - Equity peak
   - Current drawdown percentage
   - Trading status (ACTIVE/REDUCED/DISABLED)

2. **Portfolio Exposure**
   - Number of open positions
   - Total exposure amount and percentage

3. **Risk Metrics**
   - Total risk deployed (sum of all stop distances)
   - Worst-case loss (if all stops hit simultaneously)
   - Largest position (symbol and value)

4. **Recent Alerts**
   - Last 5 alerts (warnings, circuit breaker activations, etc.)

**Implementation**:
- `generate_daily_risk_report()` in `kill_switch.py:572-631`
- `format_daily_risk_report()` formats as text in `kill_switch.py:634-685`

**Example Output**:
```
============================================================
Daily Risk Report - 2026-01-21 09:00:00
============================================================

Account Status:
  Balance:           $100,000.00
  Equity Peak:       $105,000.00
  Current Drawdown:  4.76%
  Trading Status:    ACTIVE

Portfolio Exposure:
  Open Positions:    3
  Total Exposure:    $45,000.00 (45.0%)

Risk Metrics:
  Total Risk:        $2,250.00
  Worst-Case Loss:   $2,250.00
  Largest Position:  SPY ($20,000.00)

Recent Alerts:
  - INFO: Position SPY opened with 1% risk
============================================================
```

**Schedule**: Generated daily at market open (9:00 AM EST) and close (4:00 PM EST)

**Storage**: Reports saved to SQLite database and optionally exported to CSV

---

## Configuration

### PositionLimitsConfig

**Location**: `position_limits.py:69-85`

**Default Values**:
```python
PositionLimitsConfig(
    # Position-level (R11.1)
    max_risk_pct_trend=0.01,           # 1% for trend-following
    max_risk_pct_mean_rev=0.0075,      # 0.75% for mean reversion
    max_position_size_pct=0.20,        # 20% of account per position

    # Portfolio-level (R11.2)
    max_total_exposure_pct=0.50,       # 50% total deployed
    max_open_positions=5,              # Max 5 positions
    correlation_threshold=0.7,         # Reduce size if correlation > 0.7
    correlation_size_reduction=0.50,   # Reduce by 50%

    # ATR stop multipliers (R11.1.4)
    trend_stop_atr_multiplier=2.5,     # 2.5× ATR for trend
    mean_rev_stop_pct=0.03,            # 3% fixed for mean-rev
)
```

**Customization**: Create custom config for different risk profiles:

```python
# Conservative configuration
conservative_config = PositionLimitsConfig(
    max_risk_pct_trend=0.005,          # 0.5% risk
    max_risk_pct_mean_rev=0.005,       # 0.5% risk
    max_position_size_pct=0.15,        # 15% max position
    max_total_exposure_pct=0.40,       # 40% max exposure
    max_open_positions=3,              # Max 3 positions
)

# Aggressive configuration (not recommended)
aggressive_config = PositionLimitsConfig(
    max_risk_pct_trend=0.02,           # 2% risk
    max_risk_pct_mean_rev=0.015,       # 1.5% risk
    max_position_size_pct=0.25,        # 25% max position
    max_total_exposure_pct=0.75,       # 75% max exposure
)
```

---

### DrawdownCircuitBreakerConfig

**Location**: `kill_switch.py:69-82`

**Default Values**:
```python
DrawdownCircuitBreakerConfig(
    # Drawdown thresholds (R11.3)
    warning_drawdown_pct=0.15,         # 15% warning
    critical_drawdown_pct=0.20,        # 20% critical

    # Actions at warning level (R11.3.2)
    warning_size_reduction=0.50,       # Reduce sizes 50%
    warning_stop_multiplier=2.0,       # Tighten stops to 2.0× ATR
    normal_stop_multiplier=2.5,        # Normal stops at 2.5× ATR

    # Manual review (R11.3.3)
    require_manual_review=True,        # Require acknowledgment
)
```

**Customization**: Adjust thresholds based on risk tolerance:

```python
# More lenient circuit breakers (higher risk)
lenient_config = DrawdownCircuitBreakerConfig(
    warning_drawdown_pct=0.20,         # 20% warning
    critical_drawdown_pct=0.25,        # 25% critical
)

# Stricter circuit breakers (lower risk)
strict_config = DrawdownCircuitBreakerConfig(
    warning_drawdown_pct=0.10,         # 10% warning
    critical_drawdown_pct=0.15,        # 15% critical
)
```

---

## Examples

### Example 1: Opening a New Position

```python
from fluxhero.backend.risk.position_limits import (
    validate_new_position, calculate_atr_stop_loss,
    StrategyType, RiskCheckResult, Position
)

# Account and market data
account_balance = 100000.0
entry_price = 50.0
atr = 2.0
strategy_type = StrategyType.TREND_FOLLOWING

# Calculate stop loss
stop_loss = calculate_atr_stop_loss(
    entry_price=entry_price,
    atr=atr,
    side=1,  # Long
    strategy_type=strategy_type
)
# stop_loss = 45.0 (50 - 2.5×2.0)

# Calculate position size
from position_limits import calculate_position_size_from_risk
shares = calculate_position_size_from_risk(
    account_balance=account_balance,
    entry_price=entry_price,
    stop_loss=stop_loss,
    strategy_type=strategy_type
)
# shares = 200.0 (risk $1000 / price risk $5)

# Validate position
open_positions = []  # No existing positions
result, reason, adjusted_shares = validate_new_position(
    account_balance=account_balance,
    entry_price=entry_price,
    stop_loss=stop_loss,
    shares=shares,
    strategy_type=strategy_type,
    open_positions=open_positions
)

if result == RiskCheckResult.APPROVED:
    print(f"Trade approved: {adjusted_shares} shares at ${entry_price}")
else:
    print(f"Trade rejected: {reason}")
```

---

### Example 2: Portfolio Exposure Check

```python
from fluxhero.backend.risk.position_limits import Position, validate_portfolio_level_risk

# Existing positions
open_positions = [
    Position("SPY", 100, 400.0, 410.0, 395.0),   # $41,000 value
    Position("QQQ", 50, 300.0, 305.0, 291.0),     # $15,250 value
]

# New position
new_position_value = 15000.0

# Check portfolio risk
account_balance = 100000.0
result, reason = validate_portfolio_level_risk(
    account_balance=account_balance,
    open_positions=open_positions,
    new_position_value=new_position_value
)

# Current: $56,250 + new $15,000 = $71,250
# Exceeds 50% limit ($50,000)
# result = REJECTED_TOTAL_EXPOSURE
```

---

### Example 3: Drawdown Circuit Breaker

```python
from fluxhero.backend.risk.kill_switch import (
    EquityTracker, DrawdownCircuitBreaker, TradingStatus
)

# Initialize tracking
tracker = EquityTracker(initial_equity=100000)
breaker = DrawdownCircuitBreaker()

# Simulate losses
tracker.update_equity(95000)  # 5% drawdown - NORMAL
tracker.update_equity(90000)  # 10% drawdown - NORMAL
tracker.update_equity(85000)  # 15% drawdown - WARNING

# Update circuit breaker
drawdown_pct = tracker.calculate_drawdown_pct()
status, alerts = breaker.update_trading_status(drawdown_pct)

print(f"Status: {status.name}")  # REDUCED
print(f"Alerts: {alerts}")
# ["WARNING: 15% drawdown reached. Position sizes reduced 50%..."]

# Check if new trades allowed
can_trade, reason = breaker.can_open_new_position()
# (True, "Trading allowed") - but at 50% size

# Get position size multiplier
multiplier = breaker.get_position_size_multiplier()
# 0.5 (reduce to 50%)

# Further losses trigger kill switch
tracker.update_equity(80000)  # 20% drawdown - CRITICAL
drawdown_pct = tracker.calculate_drawdown_pct()
status, alerts = breaker.update_trading_status(drawdown_pct)

print(f"Status: {status.name}")  # DISABLED
should_close_all = breaker.should_close_all_positions()
# True - close all positions immediately

can_trade, reason = breaker.can_open_new_position()
# (False, "Trading disabled due to 20% drawdown...")
```

---

### Example 4: Daily Risk Report

```python
from fluxhero.backend.risk.kill_switch import (
    generate_daily_risk_report, format_daily_risk_report,
    Position, EquityTracker, DrawdownCircuitBreaker
)

# Setup
tracker = EquityTracker(100000)
tracker.update_equity(105000)  # New peak
tracker.update_equity(100000)  # Current equity
breaker = DrawdownCircuitBreaker()

# Positions
positions = [
    Position("SPY", 100, 400.0, 410.0, 395.0),
    Position("QQQ", 50, 300.0, 305.0, 291.0),
]

# Generate report
report = generate_daily_risk_report(
    account_balance=100000,
    equity_tracker=tracker,
    open_positions=positions,
    circuit_breaker=breaker
)

# Format and print
text = format_daily_risk_report(report)
print(text)
```

---

## Testing and Validation

### Unit Tests

**Location**: `tests/test_risk_position_limits.py`, `tests/test_risk_kill_switch.py`

**Coverage**:
- Position sizing calculations
- Risk limit validation
- Correlation calculations
- Drawdown tracking
- Circuit breaker triggers
- Alert generation

**Run Tests**:
```bash
pytest tests/test_risk_position_limits.py -v
pytest tests/test_risk_kill_switch.py -v
```

---

### Validation Scenarios

**Scenario 1: 5 Consecutive Losses**
- Expected: Drawdown < 6% (5 × 1% + slippage)
- Result: No circuit breaker triggered
- Pass Condition: All positions sized correctly with 1% risk

**Scenario 2: 15% Drawdown**
- Expected: Position sizes cut 50%, stops tightened to 2.0× ATR
- Result: Trading continues in REDUCED mode
- Pass Condition: New positions at 50% size, alert generated

**Scenario 3: 20% Drawdown**
- Expected: All positions closed, trading disabled
- Result: Kill switch activates
- Pass Condition: Cannot open new positions, manual review required

**Scenario 4: High Correlation**
- Expected: Position size reduced 50% when correlation > 0.7
- Result: New position opened at reduced size
- Pass Condition: Maintains diversification

**Scenario 5: Worst Market Period**
- Expected: Max drawdown < 25% during 2008/2020 crashes
- Result: System survives historical stress tests
- Pass Condition: Backtest demonstrates resilience

---

## Best Practices

### 1. Never Bypass Risk Controls

Risk controls exist to protect capital. Bypassing them removes critical safety nets.

**Don't**:
```python
# Bypassing stop loss validation
validate_position_level_risk(..., stop_loss=None)  # NEVER DO THIS
```

**Do**:
```python
# Always provide stop loss
stop_loss = calculate_atr_stop_loss(...)
validate_position_level_risk(..., stop_loss=stop_loss)
```

---

### 2. Monitor Drawdown Daily

Check drawdown percentage at market open and close.

```python
# Daily monitoring
tracker.update_equity(current_equity)
drawdown_pct = tracker.calculate_drawdown_pct()

if drawdown_pct > 0.10:
    print(f"WARNING: Drawdown at {drawdown_pct*100:.1f}%")
```

---

### 3. Review Circuit Breaker Alerts

When circuit breakers trigger, investigate immediately.

**Questions to Ask**:
- What caused the losses?
- Were the signals valid?
- Did market conditions change?
- Should strategy parameters be adjusted?
- Is the market regime different than expected?

---

### 4. Respect Manual Review Requirements

When the kill switch activates, do NOT rush to resume trading.

**Manual Review Checklist**:
- [ ] Review all closed trades and understand losses
- [ ] Check if strategy assumptions are still valid
- [ ] Verify market conditions have improved
- [ ] Adjust risk parameters if needed
- [ ] Confirm emotional state is stable (not revenge trading)
- [ ] Document lessons learned
- [ ] Only then: `acknowledge_manual_review()`

---

### 5. Test Configuration Changes

Before deploying new risk configurations, backtest thoroughly.

```python
# Test new configuration on historical data
new_config = PositionLimitsConfig(max_risk_pct_trend=0.02)

# Run backtest with new config
results = run_backtest(config=new_config)

# Validate max drawdown is acceptable
assert results.max_drawdown < 0.25  # Still under 25%
```

---

## Summary

The FluxHero Risk Management System provides comprehensive protection through:

1. **Position-Level Controls**: Limit risk per trade (1% trend, 0.75% mean-rev) and position size (20%)
2. **Portfolio-Level Controls**: Limit total exposure (50%) and number of positions (5)
3. **Correlation Monitoring**: Reduce position sizes when correlation > 0.7
4. **Drawdown Circuit Breakers**: Automatic risk reduction at 15%, kill switch at 20%
5. **Real-Time Monitoring**: Continuous visibility into risk metrics
6. **Daily Reporting**: Comprehensive daily risk summaries

**Key Principle**: Protect capital first, generate returns second. Risk management is not optional.

**Remember**: The goal is not to avoid all losses (impossible), but to ensure losses are small, controlled, and survivable while allowing winners to run.

---

**Last Updated**: 2026-01-21
**Version**: 1.0
**Author**: FluxHero Development Team
