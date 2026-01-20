# Algorithmic Trading System Guide
*A brutally honest guide to building quantitative trading systems*

## Table of Contents
1. [Core Components You Need](#core-components-you-need)
2. [Common Pitfalls](#common-pitfalls)
3. [Understanding Futures](#understanding-futures)
4. [Market-Specific Analysis](#market-specific-analysis)
5. [Strategy Ideas](#strategy-ideas)
6. [Risk Management](#risk-management)
7. [Implementation Priorities](#implementation-priorities)
8. [Realistic Expectations](#realistic-expectations)

---

## Core Components You Need

### 1. Market Microstructure
- **Order execution**: Market vs limit orders, order book dynamics
- **Slippage**: Difference between backtest price and actual execution price
- **Spread costs**: Every trade costs you the bid-ask spread (you buy at ask, sell at bid)
- **Market impact**: Your orders move the price, especially with size

### 2. Data & Backtesting
- **Historical data quality**: Survivorship bias, look-ahead bias, corporate actions all corrupt backtests
- **Your backtest is probably lying**: Overfitting is the default, not the exception
- **Out-of-sample testing**: Mandatory, not optional
- **Transaction costs**: Will kill most strategies that look good on paper

### 3. Execution System
- Order types and when to use them
- Latency matters (even for "slower" strategies)
- Partial fills, rejections, and all the ways orders can fail
- Position sizing and risk management (where most people blow up)

---

## Common Pitfalls

### The Overfitting Trap
You'll find patterns in random noise. Every parameter you optimize, every indicator you add, increases the chance your strategy is just curve-fitted garbage. Most "profitable" backtests are just sophisticated data mining.

**Solution**: Use walk-forward testing, out-of-sample validation, and keep strategies simple.

### Transaction Costs
Spreads, commissions, fees, slippage add up fast. A strategy that makes 0.1% per trade looks amazing until you realize you're paying 0.08% in costs.

**Reality check**:
- Crypto: 0.05-0.1% per side (maker/taker fees)
- Stocks: $0.005/share + spread
- Futures: $1-5 per contract + spread

### Regime Changes
Markets change. A strategy that worked 2015-2020 might be useless now. Mean reversion worked great until it didn't. Momentum works until it reverses violently.

**Solution**: Test across multiple market regimes (bull, bear, sideways, high vol, low vol).

### Liquidity Assumptions
Your backtest assumes you can trade any size at the mid price. Reality: you're trading at the ask and selling at the bid, and if you're moving size, you're moving the price against yourself.

**Solution**: Model realistic slippage based on actual order book depth.

### Live Trading Psychology
Even with algos, you'll want to turn it off during drawdowns. You'll want to "improve" it after losses. Discipline is harder than the math.

**Solution**: Set rules BEFORE going live. Maximum drawdown triggers automatic pause, not discretionary decisions.

---

## Understanding Futures

### What Are Futures?
Agreement to buy/sell an asset at a specific price on a specific date. In practice, especially in crypto/stock index futures, almost nobody takes delivery - you just settle the profit/loss in cash.

### Why Futures Exist
- **Leverage**: Control $100k of Bitcoin with $10k (10x leverage)
- **Hedging**: Lock in prices, protect positions
- **Short selling**: Easier than borrowing spot assets
- **24/7 trading**: Crypto futures trade continuously

### Crypto Futures Types

#### Perpetual Futures (Perps)
- No expiration date
- Tracks spot price through "funding rates"
- **Funding rate**: Periodic payment between longs and shorts (every 8 hours typically)
  - If futures > spot: longs pay shorts
  - If futures < spot: shorts pay longs
- Popular on: Binance, Bybit, OKX, dYdX

**Critical**: During bull runs, funding rates can be 0.1-0.3% per 8 hours (0.3-0.9% per day = 100-300% annualized). This will destroy your returns.

#### Traditional (Quarterly) Futures
- Expire every quarter (March, June, September, December)
- Settle to spot price at expiration
- Popular on: CME (regulated), Deribit

### Stock Index Futures
- **ES** (S&P 500), **NQ** (Nasdaq) - not individual stock futures
- Better leverage than ETFs
- Better tax treatment in US (60/40 long-term/short-term split)
- Nearly 24-hour trading
- Better fills for large size

### Where to Trade

**Crypto Futures**:
- Binance Futures - highest volume
- Bybit - popular for perps
- dYdX - decentralized
- CME - regulated, institutional

**Stock Index Futures**:
- Interactive Brokers - best for retail algo traders
- TradeStation - decent for futures
- TD Ameritrade/Thinkorswim - higher fees

### The Leverage Trap

**This will kill you if you're not careful:**

With **10x leverage**, a 10% move against you = **100% loss** (liquidation).

Bitcoin regularly moves 10% in a day. You can be completely right on direction and still get liquidated on volatility.

**Rule**: Use 2-5x leverage max, not 10-100x. If you'd trade $10k spot, trade $1k with 10x leverage (same exposure).

### Spot vs Futures Comparison

| Factor | Spot | Futures |
|--------|------|---------|
| Ownership | You own the asset | Contract only |
| Liquidation risk | None | Yes - automatic closure |
| Funding costs | None | Yes (perps) or rollover (quarterly) |
| Shorting | Complex | Easy |
| Leverage | Limited | High |
| Holding period | Unlimited | Perps: unlimited with costs<br>Quarterly: must roll |

---

## Market-Specific Analysis

### Crypto Day Trading
- **Difficulty**: Extreme
- **Volatility**: 10% moves in minutes are normal
- **Costs**: Funding costs kill you on perps
- **Risks**: Exchange outages, liquidations, manipulation
- **Edge**: Exploiting retail panic/FOMO, but competing against bots
- **Verdict**: Possible but high risk. Start small, expect losses while learning.

### Crypto Swing Trading
- **Difficulty**: High
- **Volatility**: Double-edged sword (big gains, big losses)
- **Costs**: Funding costs brutal for multi-day holds on perps
- **Risks**: Fast regime changes (momentum → mean reversion → momentum)
- **Verdict**: More viable than day trading. Use spot or low leverage. Accept 20% drawdowns.

### Stock Day Trading
- **Difficulty**: Very High
- **Competition**: HFT firms, professional traders
- **Edge**: Extremely thin after costs
- **Requirements**: $25k minimum (PDT rule in US)
- **Verdict**: Hardest game. Most retail day traders lose. Need genuine edge.

### Stock Swing Trading
- **Difficulty**: Moderate
- **Research**: Plenty of academic backing (momentum, value, quality factors)
- **Competition**: Most factors are crowded now
- **Verdict**: Most realistic for retail algo trading. Focus on risk management.

### Futures Day Trading (ES/NQ)
- **Difficulty**: Extreme
- **Competition**: Prop firms with microsecond edge
- **Leverage**: Amplifies small edges AND mistakes
- **Costs**: Commissions add up quickly
- **Verdict**: Unless you have HFT infrastructure, you'll struggle.

### Futures Swing Trading
- **Difficulty**: Moderate-High
- **Costs**: Rollover costs, margin requirements
- **Liquidity**: Excellent (ES, NQ, CL)
- **Leverage**: Can use conservatively (2-3x) for capital efficiency
- **Verdict**: Viable for trend-following, momentum strategies. Watch rollover dates.

---

## Strategy Ideas

### Day Trading Strategies

#### 1. Mean Reversion on Volatility Spikes
- **Concept**: When crypto moves 5%+ in 30min, fade the move
- **Why it works**: Overreaction, liquidation cascades, then bounce
- **Why it fails**: Sometimes the move is the start of a bigger trend
- **Implementation**: Real-time data, fast execution, tight stops

#### 2. Opening Range Breakout (Stocks)
- **Concept**: Mark high/low of first 15-30min, trade breakouts
- **Why it works**: Day traders establish direction early
- **Why it fails**: Extremely crowded, whipsaws common
- **Implementation**: Need filters (volume, gap size, sector momentum)

#### 3. News Fade (Advanced)
- **Concept**: Fade initial reaction after move exhausts
- **Why it works**: Overreaction to headlines, algos amplify
- **Why it fails**: Sometimes news is legitimately huge
- **Implementation**: Hard to automate (sentiment analysis)

### Swing Trading Strategies

#### 1. Momentum Continuation
- **Concept**: Buy assets making new 20-day highs with volume
- **Why it works**: Momentum persists short-term, FOMO kicks in
- **Why it fails**: Violent reversals (2021 meme stocks)
- **Implementation**: Filter out low liquidity, scale in/out

#### 2. Mean Reversion to Moving Averages
- **Concept**: Buy when price touches 20-day MA in uptrend
- **Why it works**: Institutions use MAs as support/resistance
- **Why it fails**: Trends break, MAs lag
- **Implementation**: Combine with RSI/volume filters

#### 3. Volatility Breakout
- **Concept**: When ATR expands, trade direction of breakout
- **Why it works**: Low volatility clusters, then explodes
- **Why it fails**: False breakouts in ranging markets
- **Implementation**: Multi-timeframe confirmation, tight stops

#### 4. Pair Trading / Statistical Arbitrage
- **Concept**: Trade spread between correlated assets (BTC/ETH, SPY/QQQ)
- **Why it works**: Correlations mean-revert, less directional risk
- **Why it fails**: Correlations break during regime changes
- **Implementation**: Cointegration testing, careful position sizing

---

## Risk Management

### 20% Maximum Drawdown Guidelines

**What this means**:
- Most institutional funds aim for 10-15% max
- 20% is aggressive but doable with discipline
- Individual position risk: **1-1.5% of capital per trade max**

**For a $100k account**:
- Max risk per trade: $1,000-1,500
- With 5x leverage on futures: control $500k exposure, but stop loss tight enough to only lose $1k
- Most people screw this up - they use full leverage without proper stops

### Critical Risk Management Rules

1. **Never risk more than 1.5% per trade**
   - Assume 15-trade losing streak is possible
   - 1.5% × 15 = 22.5% drawdown (close to your 20% limit)

2. **Position sizing for volatility**
   - BTC has 5x volatility of SPY
   - Use ATR-based position sizing, not fixed %
   - If you risk 1% on SPY with 2% stop, you need 10% stop on BTC

3. **Max exposure limits**
   - Never be >50% invested across all positions
   - Don't have 10 correlated positions (that's 10x concentrated risk)
   - Diversify across assets and strategies

4. **Drawdown circuit breakers**
   - At 15% drawdown: reduce position sizes by 50%
   - At 20% drawdown: stop trading, reassess everything
   - These are hard rules, not suggestions

5. **Correlation monitoring**
   - Track correlation between positions
   - Multiple BTC positions = same as one large BTC position
   - Diversification only works if assets aren't correlated

---

## Implementation Priorities

Build in this exact order:

### 1. Risk Management System (FIRST, not last)
- Position sizing calculator
- Drawdown tracker (kill strategy at 15%)
- Correlation monitor
- Max exposure limits

### 2. Backtesting Engine
- Realistic slippage model (buy at ask, sell at bid)
- Transaction costs:
  - Crypto: 0.05-0.1% per side
  - Stocks: $0.005/share
  - Futures: $1-5/contract
- Funding costs for crypto perps (use historical data)
- Walk-forward testing (train period A, test period B, repeat)

### 3. Data Pipeline
**Day trading**: Tick/minute data, real-time feeds
**Swing trading**: Daily/hourly data sufficient

**Sources**:
- Stocks: Alpha Vantage, Polygon.io, IB API
- Crypto: Binance API, CoinGecko, Deribit
- Futures: IB API, CQG, Norgate Data

### 4. Execution System
- Order management (submit, modify, cancel)
- Fill tracking (partial fills, rejections)
- Failover (connection drops?)
- Day trading: sub-second execution needed
- Swing trading: 5-10 second delays tolerable

### 5. Monitoring & Alerts
- Real-time P&L tracking
- Drawdown alerts (notify at >10%)
- Position tracking (always know what you own)
- Performance vs backtest (investigate if >10% deviation)

---

## Realistic Expectations

### Day Trading Performance
- **Great year**: 20-30% annual return, Sharpe 1.2-1.5
- **Good year**: 10-15% return, Sharpe 0.8-1.0
- **Realistic**: 5-10% return, Sharpe 0.5-0.8
- **Likely**: Break-even after costs
- **Max drawdown**: 15-25% (even good traders see this)

**Win rate**: 45-55% typical
**Risk/reward**: Average win needs to be >1.5x average loss

### Swing Trading Performance
- **Great year**: 30-50% return, Sharpe 1.5-2.0
- **Good year**: 15-25% return, Sharpe 1.0-1.3
- **Realistic**: 10-15% return, Sharpe 0.7-1.0
- **Likely**: 5-10% return (better than day trading)
- **Max drawdown**: 20-30% (expect this even with good strategy)

**Win rate**: 40-50% typical
**Risk/reward**: Need 2:1+ risk/reward to be profitable

### Red Flags (Probably Lying/Overleveraged/Ponzi)
- Consistent 100%+ annual returns
- Sharpe ratio >3
- <10% max drawdowns with high returns
- "Always profitable" months

If it sounds too good to be true, it is.

---

## Specific Pitfalls for Day/Swing Trading

### 1. Overfitting on Recent Regime
Your 2020-2023 backtest includes:
- Extreme bull (2020-2021)
- Bear market (2022)
- Recovery (2023-2024)

Strategy that worked in one regime will fail in another.

**Test across**: 2015-2024 to capture multiple regimes.

### 2. Ignoring Overnight Gaps
Swing trading = overnight exposure.

- Stocks gap 5-10% on earnings
- Crypto gaps on hacks, regulatory news
- Your backtest needs daily open/close data, not just continuous prices

### 3. Position Sizing for Volatility
Don't use fixed % stops across different assets.

- SPY 2% stop = reasonable
- BTC 2% stop = getting stopped out on noise

Use ATR-based stops (e.g., 2× ATR).

### 4. Funding Costs on Crypto Perps
Holding BTC perps for 5 days during bull run:
- 0.3% per day × 5 days = 1.5% cost
- That's 30% annualized just to hold

Your edge must be > funding costs, or use spot/quarterly futures.

### 5. Win Rate vs Risk/Reward Trap
- Day traders: 60% win rate but lose money (small wins, big losses)
- Swing traders: 40% win rate but make money (small losses, big wins)

Track your own stats and adjust strategy accordingly.

---

## Recommended Starting Path

### Phase 1: Choose Your Focus
**Pick ONE market to start**:
- **Crypto swing**: Higher volatility, 24/7, less capital needed
- **Stock swing**: Regulated, better data, more research available

**Do NOT**:
- Try both simultaneously
- Start with day trading (too hard)
- Use leverage initially

### Phase 2: Build Infrastructure (3-6 months)
1. Risk management system
2. Backtesting engine
3. Data pipeline
4. Validate with simple strategies (MA crossover, RSI)

### Phase 3: Paper Trading (3-6 months)
1. Trade with spot (no leverage)
2. Learn execution challenges
3. Build monitoring systems
4. Prove strategy works live

### Phase 4: Live Trading - Small Size (6-12 months)
1. Start with 10-25% of intended capital
2. Track live P&L vs backtest (should match within 10-20%)
3. Measure: win rate, avg win/loss, Sharpe, drawdown
4. Cost per trade analysis

### Phase 5: Scale & Add Leverage (Only if profitable)
1. After 6+ months profitable trading
2. Add 2x leverage max initially
3. Monitor impact on drawdowns
4. Never exceed 3-5x even if profitable

---

## Key Metrics to Track Obsessively

### Performance Metrics
- **Win rate**: % of profitable trades
- **Average win/loss**: Risk/reward ratio
- **Sharpe ratio**: Risk-adjusted returns (>1 is good)
- **Max drawdown**: Worst peak-to-trough loss
- **Max consecutive losses**: Stress test your psychology

### Execution Metrics
- **Live P&L vs backtest**: Should match within 10-20%
- **Slippage per trade**: How much worse than backtest?
- **Cost per trade**: Fees eating you alive?
- **Fill rate**: How often do orders get filled?

### Risk Metrics
- **Current drawdown**: Distance from equity peak
- **Position correlation**: Are you actually diversified?
- **Exposure**: % of capital deployed
- **Leverage**: Actual leverage vs intended

---

## Final Brutal Truth

Most retail traders lose money not because their strategy is wrong, but because:

1. **Overleveraging** - using 10x+ when they should use 2-3x
2. **Ignoring costs** - funding rates and fees destroy profitability
3. **Getting liquidated** - right on direction, wrong on timing/volatility
4. **Poor risk management** - no stop losses, no position sizing
5. **Lack of discipline** - turning off strategy during drawdowns

**The edge isn't in finding the magic indicator.**

The edge is in:
- Not overfitting
- Properly accounting for costs
- Managing risk so you survive drawdowns
- Having discipline to stick with it through losing streaks

A mediocre strategy with excellent risk management beats a "perfect" strategy with sloppy implementation.

---

## Data Sources Reference

### Stock Market Data
- **Alpha Vantage**: Free tier available, good for starting
- **Polygon.io**: Professional data, reasonable pricing
- **Interactive Brokers API**: If you trade with them
- **Yahoo Finance**: Free but quality concerns

### Crypto Data
- **Binance API**: Free, excellent for spot and futures
- **CoinGecko API**: Free tier, good for historical data
- **Deribit API**: Options and futures data
- **Glassnode**: On-chain metrics (paid)

### Futures Data
- **Interactive Brokers API**: Good for index futures
- **CQG**: Professional grade (expensive)
- **Norgate Data**: End-of-day data (affordable)

### Funding Rate Data (Crypto Perps)
- Most exchanges provide historical funding rates via API
- Critical for backtesting perpetual futures strategies
- Binance, Bybit APIs have this data

---

*Last updated: 2026-01-18*
*Remember: The market is always right. Your backtest is probably wrong.*
