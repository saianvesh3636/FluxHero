# Quantitative Trading Guide: Q&A Reference

A comprehensive guide covering technical indicators, smoothing, regression, and adaptive strategies.

---

## Table of Contents

1. [EMA (Exponential Moving Average)](#1-ema-exponential-moving-average)
2. [RSI (Relative Strength Index)](#2-rsi-relative-strength-index)
3. [MACD (Moving Average Convergence Divergence)](#3-macd-moving-average-convergence-divergence)
4. [True Range & ATR](#4-true-range--atr)
5. [Data Transformations](#5-data-transformations)
6. [The Three Types of "Alpha"](#6-the-three-types-of-alpha)
7. [When to Use What in Trading](#7-when-to-use-what-in-trading)
8. [Volatility-Based Trading](#8-volatility-based-trading)
9. [Adaptive Scaling with Regression](#9-adaptive-scaling-with-regression)
10. [Quantile Regression](#10-quantile-regression)

---

## 1. EMA (Exponential Moving Average)

### What is EMA?

EMA is a moving average that gives **more weight to recent prices**. Unlike SMA which treats all prices equally, EMA reacts faster to recent price changes.

### Formula

```
EMA_today = (Price_today × α) + (EMA_yesterday × (1 - α))
```

Where:
```
α (Multiplier) = 2 ÷ (Period + 1)
```

### Example: 5-Day EMA

| Day | Price | Calculation | EMA |
|-----|-------|-------------|-----|
| 1-5 | Various | SMA = (10+11+12+11+13)/5 | 11.40 |
| 6 | $14 | (14 × 0.333) + (11.40 × 0.667) | 12.27 |

### Why α Must Be Between 0 and 1

1. **Weights must sum to 1**: α + (1-α) = 1
2. **Both weights must be positive**: Can't have negative influence
3. **Boundary meanings**:
   - α = 0 → Only trust past (no new data)
   - α = 1 → Only trust current (no smoothing)

---

## 2. RSI (Relative Strength Index)

### What is RSI?

RSI measures the **speed and magnitude of price changes** to identify overbought/oversold conditions. Range: 0-100.

- **RSI > 70** → Overbought (price may drop)
- **RSI < 30** → Oversold (price may rise)

### Formula

```
RSI = 100 - (100 ÷ (1 + RS))
RS = Average Gain ÷ Average Loss
```

### Example

| Day | Change | Gain | Loss |
|-----|--------|------|------|
| 1 | +0.50 | 0.50 | 0 |
| 2 | -0.75 | 0 | 0.75 |
| 3 | +0.75 | 0.75 | 0 |

```
Avg Gain = 0.40, Avg Loss = 0.20
RS = 0.40 / 0.20 = 2.0
RSI = 100 - (100/3) = 66.67
```

---

## 3. MACD (Moving Average Convergence Divergence)

### Components

| Component | Calculation |
|-----------|-------------|
| MACD Line | 12-period EMA - 26-period EMA |
| Signal Line | 9-period EMA of MACD Line |
| Histogram | MACD Line - Signal Line |

### Trading Signals

| Signal | Meaning |
|--------|---------|
| MACD crosses ABOVE Signal | Bullish (buy) |
| MACD crosses BELOW Signal | Bearish (sell) |
| Histogram growing | Momentum increasing |
| Histogram shrinking | Momentum weakening |

---

## 4. True Range & ATR

### What is True Range?

True Range captures the **complete price movement** including gaps between sessions.

### Formula

```
TR = MAX of:
  1. High - Low
  2. |High - Previous Close|
  3. |Low - Previous Close|
```

### Why Three Methods?

| Scenario | Winning Formula |
|----------|-----------------|
| Normal day (no gap) | High - Low |
| Gap UP | High - Previous Close |
| Gap DOWN | Previous Close - Low |

### ATR (Average True Range)

```
ATR = Moving average of True Range over N periods
```

### Common Uses

1. **Stop-Loss**: `Stop = Entry - (ATR × 2)`
2. **Position Sizing**: `Shares = Risk $ ÷ ATR`
3. **Breakout Detection**: `Signal when Price > Close + (ATR × 1.5)`

---

## 5. Data Transformations

### Why Transform Data?

To fix:
- Non-linear relationships → Make them linear
- Non-constant variance → Stabilize it
- Skewed distributions → Normalize them

### Common Transformations

| Transformation | Formula | Use When |
|----------------|---------|----------|
| Reciprocal | 1/x | Hyperbolic curves, rates |
| Log | log(x) | Exponential data, wide ranges |
| Square Root | √x | Count data, moderate skew |
| Square | x² | Left-skewed data |

### Example: Reciprocal

**Problem**: Time vs Speed is curved (Time = Distance/Speed)

**Solution**: Use 1/Speed → Relationship becomes linear

---

## 6. The Three Types of "Alpha"

### Alpha (Returns) - Performance Measure

**Meaning**: Extra return earned beyond market performance

```
Alpha = Your Return - Market Return
```

| Alpha | Interpretation |
|-------|----------------|
| +5% | Beat market by 5% |
| 0% | Matched market |
| -3% | Underperformed by 3% |

**"Alpha of 1"** = Beat the market by 1%

### Alpha (Smoothing) - EMA Parameter

**Meaning**: Weight given to most recent data in EMA

- α close to 1 → More reactive
- α close to 0 → More smooth

### Alpha (Strategy) - Trading Idea

**Meaning**: The trading edge or strategy itself

Examples: Momentum alpha, Mean reversion alpha, News alpha

### Memory Trick (Restaurant Analogy)

| Alpha Type | Analogy |
|------------|---------|
| Strategy | The secret recipe |
| Smoothing | How much salt to add |
| Returns | Your profit at month end |

---

## 7. When to Use What in Trading

### By Trading Style

| Style | EMA Period | RSI Period | ATR Period |
|-------|------------|------------|------------|
| Day Trading | 9, 21 | 7-9 | 10 |
| Swing Trading | 12, 26 | 14 | 14 |
| Position Trading | 50, 200 | 21 | 20 |

### By Market Condition

| Condition | Use | Avoid |
|-----------|-----|-------|
| Trending | EMA crossovers, MACD | RSI overbought/oversold |
| Ranging | RSI, support/resistance | EMA crossovers |
| High Volatility | Trend following | Mean reversion |
| Low Volatility | Mean reversion | Trend following |

### Decision Matrix

| Your Goal | Use This |
|-----------|----------|
| Trend direction | EMA, SMA |
| Overbought/oversold | RSI |
| Momentum shifts | MACD |
| Stop-loss distance | ATR |
| Position sizing | ATR |

---

## 8. Volatility-Based Trading

### What Volatility Tells Us

| State | Meaning | Action |
|-------|---------|--------|
| Low & Stable | Consolidation | Expect breakout |
| Low → Rising | Breakout starting | Follow the move |
| High & Rising | Strong trend | Wide stops, follow trend |
| High → Falling | Move exhausting | Take profits |

### Key Insight

**Volatility is context, not signal** — use it to decide HOW to trade, not just WHAT to trade.

### Volatility + Alpha Relationship

| Volatility | Best Alpha | Why |
|------------|------------|-----|
| Low | Low α | Small wiggles are noise, smooth them |
| High | High α | Big moves are real, react fast |

**Simple Rule**:
> "When the market whispers, listen to history."
> "When the market shouts, listen to NOW."

---

## 9. Adaptive Scaling with Regression

### The Goal

Find a formula: `α = m × ATR + c`

Instead of guessing alpha, let data tell you the optimal relationship.

### Components

| Term | Meaning | Plain English |
|------|---------|---------------|
| m (slope) | How much α changes per ATR unit | "For every 1% increase in volatility, increase alpha by m" |
| c (intercept) | Baseline α when ATR = 0 | "Even in calm markets, use at least c% weight on new data" |

### How to Find m and c

**Least Squares Method**: Find the line that minimizes total errors

1. For each historical day, test many alphas (0.05, 0.10, 0.15...)
2. Find which alpha worked best (smallest prediction error)
3. Record pair: (ATR for that day, Best alpha)
4. Repeat for many days
5. Run regression on all pairs to find m and c

### What is "Error"?

```
Error = |Predicted Value - Actual Value|
```

Small error = Good prediction, formula works
Large error = Bad prediction, need better formula

### Handling Non-Linear Relationships

| If Data Looks Like... | Use |
|-----------------------|-----|
| Straight line | Linear: y = mx + c |
| Curves then flattens | Log: y = a × log(x) + c |
| S-shaped | Sigmoid: y = 1/(1 + e^(-kx)) |

### Keeping α Between 0 and 1

**Method 1: Clipping** (Simple)
```python
alpha = max(0.05, min(alpha, 0.95))
```

**Method 2: Sigmoid** (Smooth, natural bounds)
```python
alpha = 1 / (1 + exp(-k * (ATR - midpoint)))
```

---

## 10. Quantile Regression

### Regular vs Quantile Regression

| Type | Finds | Use Case |
|------|-------|----------|
| Regular | Average relationship | "What's typical?" |
| Quantile | Different scenarios | "What happens in best/worst cases?" |

### Example

For ATR = 2%, quantile regression shows:

| Quantile | Alpha | Meaning |
|----------|-------|---------|
| 10th | 0.25 | Conservative/safe |
| 50th | 0.40 | Typical case |
| 90th | 0.55 | Aggressive |

### When to Use Which Quantile

| Risk Preference | Quantile |
|-----------------|----------|
| Conservative | 25th-40th |
| Balanced | 50th (median) |
| Aggressive | 60th-75th |

---

## Quick Reference Tables

### Indicator Comparison

| Indicator | Best For | Range |
|-----------|----------|-------|
| EMA | Trend following | Unlimited |
| RSI | Overbought/Oversold | 0-100 |
| MACD | Momentum + Trend | Unlimited |
| ATR | Volatility measure | 0+ |

### Smoothing Factor Selection

| α Value | Behavior | Use For |
|---------|----------|---------|
| 0.05-0.15 | Very smooth | Long-term trends |
| 0.15-0.30 | Moderate | Swing trading |
| 0.30-0.50 | Responsive | Day trading |
| 0.50-0.90 | Very fast | Scalping, volatile assets |

### Transformation Selection

| Data Pattern | Transformation |
|--------------|----------------|
| Exponential growth | Log |
| Hyperbolic/rates | Reciprocal |
| Count data | Square root |
| Need bounds [0,1] | Sigmoid |

---

## Key Takeaways

1. **EMA** reacts faster than SMA; α controls the speed
2. **RSI** works best in ranging markets; be cautious in trends
3. **MACD** combines trend and momentum information
4. **ATR** captures true volatility including gaps
5. **Alpha** has three meanings: returns, smoothing factor, and strategy
6. **Match strategy to conditions**: trend-follow in high vol, mean-revert in low vol
7. **Regression** lets data determine optimal parameters instead of guessing
8. **Error** = difference between prediction and reality; smaller is better
9. **Quantile regression** shows the range of possibilities, not just average

---

*Document created: January 19, 2026*
*Based on Q&A session covering quant trading fundamentals*
