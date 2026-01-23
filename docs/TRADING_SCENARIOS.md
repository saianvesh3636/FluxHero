# FluxHero Trading System - Scenario Walkthroughs

This document walks through realistic trading scenarios showing how data flows through the system, from Alpaca integration to order execution.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Scenario 1: System Startup & Data Initialization](#scenario-1-system-startup--data-initialization)
3. [Scenario 2: Trend-Following Long Entry](#scenario-2-trend-following-long-entry)
4. [Scenario 3: Mean Reversion Short Entry](#scenario-3-mean-reversion-short-entry)
5. [Scenario 4: Stop Loss Hit](#scenario-4-stop-loss-hit)
6. [Scenario 5: Circuit Breaker Activation](#scenario-5-circuit-breaker-activation)
7. [Scenario 6: Order Chase Logic](#scenario-6-order-chase-logic)
8. [Scenario 7: Correlation-Based Size Reduction](#scenario-7-correlation-based-size-reduction)
9. [Data Flow Diagram](#data-flow-diagram)
10. [Key Code References](#key-code-references)

---

## System Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   ALPACA API    │     │   FLUXHERO      │     │   FRONTEND      │
│                 │     │   BACKEND       │     │                 │
│  REST: Bars     │────▶│  DataPipeline   │     │  WebSocket      │
│  WS: Live Ticks │────▶│  SignalGen      │────▶│  Client         │
│  Orders API     │◀────│  OrderManager   │     │                 │
│  Account API    │────▶│  RiskManager    │     │  REST Client    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Scenario 1: System Startup & Data Initialization

**Goal**: System connects to Alpaca, fetches historical data, and begins streaming.

### Step 1: Application Startup

```
File: backend/api/server.py (lines 226-308)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize SQLite database
    await app_state.sqlite_store.initialize()

    # 2. Load test CSV data (development mode)
    # In production: DataPipeline.start() would connect to Alpaca
    app_state.test_data = load_csv_data(["SPY", "AAPL", "MSFT"])
```

### Step 2: Connect to Alpaca (Production Mode)

```
File: backend/data/fetcher.py (lines 693-733)

class DataPipeline:
    async def start(self, symbol: str) -> List[Candle]:
        # Step 2a: Fetch 500 historical candles via REST
        candles = await self.rest_client.fetch_candles(
            symbol=symbol,
            timeframe="1h",
            limit=500
        )
        # This calls: GET https://data.alpaca.markets/v2/stocks/SPY/bars

        # Step 2b: Connect to WebSocket
        await self.ws_feed.connect()
        # Connects to: wss://stream.data.alpaca.markets/v2/iex

        # Step 2c: Subscribe to symbol
        await self.ws_feed.subscribe([symbol])
        # Sends: {"action": "subscribe", "bars": ["SPY"]}

        return candles  # For initial indicator calculation
```

### Step 3: REST API Call to Alpaca

```
File: backend/data/fetcher.py (lines 291-351)

async def fetch_candles(self, symbol: str, ...) -> List[Candle]:
    url = f"{self.base_url}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Hour",
        "limit": 500,
        "adjustment": "split"
    }

    # Rate limiting: 200 requests/minute
    await self.rate_limiter.acquire()

    # Retry logic: 3 attempts with exponential backoff
    for attempt in range(3):
        try:
            response = await self.client.get(url, params=params)
            data = response.json()
            return [Candle(
                timestamp=bar["t"],
                open=bar["o"],
                high=bar["h"],
                low=bar["l"],
                close=bar["c"],
                volume=bar["v"]
            ) for bar in data["bars"]]
        except httpx.HTTPError:
            await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
```

### Step 4: WebSocket Connection

```
File: backend/data/fetcher.py (lines 494-563)

class WebSocketFeed:
    async def connect(self):
        self.ws = await websockets.connect(
            "wss://stream.data.alpaca.markets/v2/iex"
        )
        # Authenticate
        await self.ws.send(json.dumps({
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret
        }))

    async def subscribe(self, symbols: List[str]):
        await self.ws.send(json.dumps({
            "action": "subscribe",
            "bars": symbols  # Subscribe to 1-minute bars
        }))
```

### Result: System Ready

```
Account: $100,000 initial capital
Positions: 0 open
Data: 500 historical candles loaded
WebSocket: Connected, streaming SPY, AAPL, MSFT
Status: ACTIVE
```

---

## Scenario 2: Trend-Following Long Entry

**Context**: SPY is in a strong uptrend. The system detects a KAMA crossover and enters a long position.

### Step 1: Price Update Arrives

```
Alpaca WebSocket Message:
{
    "T": "b",           // Bar message
    "S": "SPY",         // Symbol
    "o": 450.00,        // Open
    "h": 451.50,        // High
    "l": 449.80,        // Low
    "c": 451.25,        // Close
    "v": 1500000,       // Volume
    "t": "2024-01-22T14:30:00Z"
}

File: backend/data/fetcher.py (lines 582-633)
async for message in self.ws_feed.stream():
    if self._signal_callback:
        asyncio.create_task(self._signal_callback(message))
```

### Step 2: Indicator Calculation

```
File: backend/computation/indicators.py

Current Values After Processing:
- Close Price: $451.25
- KAMA (20): $448.50
- ATR (14): $3.20
- RSI (14): 62
- ADX (14): 32 (Strong trend: > 25)
- R² (20): 0.81 (Strong linear trend: > 0.6)
```

### Step 3: Regime Detection

```
File: backend/strategy/regime_detector.py (lines 362-421)

def classify_trend_regime(adx: float, r_squared: float) -> int:
    # ADX = 32 > 25 ✓
    # R² = 0.81 > 0.6 ✓
    return REGIME_STRONG_TREND  # Code: 2

Regime: STRONG_TREND
Volatility: NORMAL (ATR = 3.20, within normal range)
```

### Step 4: Signal Generation (Trend-Following Mode)

```
File: backend/strategy/dual_mode.py (lines 35-123)

Entry Condition Check:
- Price ($451.25) > KAMA ($448.50) + 0.5 × ATR ($1.60)
- $451.25 > $450.10 ✓ ENTRY TRIGGERED

def generate_trend_following_signals(price, kama, atr):
    entry_threshold = kama + (0.5 * atr)  # $450.10
    if price > entry_threshold:
        return SignalType.LONG  # BUY signal

Stop Loss Calculation (lines 127-178):
- Trailing stop: Entry - (2.5 × ATR)
- Stop = $451.25 - (2.5 × $3.20) = $443.25
```

### Step 5: Signal Explanation Created

```
File: backend/strategy/signal_generator.py (lines 298-393)

SignalExplanation:
    signal_type: LONG (1)
    symbol: "SPY"
    price: 451.25
    timestamp: 2024-01-22T14:30:00Z

    strategy_mode: "TREND"
    regime: "STRONG_TREND"
    volatility_state: "NORMAL"

    atr: 3.20
    kama: 448.50
    rsi: 62
    adx: 32
    r_squared: 0.81

    risk_amount: $1,000 (1% of $100,000)
    risk_percent: 0.01
    stop_loss: 443.25
    position_size: 125 shares

    entry_trigger: "KAMA crossover (Price > KAMA + 0.5×ATR)"
    noise_filter_passed: True
    volume_validated: True

Formatted Reason:
"BUY SPY @ $451.25
 Reason: KAMA crossover (Price $451.25 > KAMA+0.5×ATR $450.10)
 Regime: STRONG_TREND (ADX=32, R²=0.81)
 Risk: $1,000 (1% account), Stop: $443.25"
```

### Step 6: Risk Validation (5-Point Check)

```
File: backend/risk/position_limits.py

CHECK 1: Position-Level Risk (lines 160-226)
┌─────────────────────────────────────────────────┐
│ ✓ Stop loss set: $443.25                        │
│ ✓ Risk per trade: $1,000 (1%) ≤ 1% limit       │
│ ✓ Position value: $56,406 (11.3%) ≤ 20% limit  │
└─────────────────────────────────────────────────┘

CHECK 2: Portfolio-Level Risk (lines 277-326)
┌─────────────────────────────────────────────────┐
│ ✓ Open positions: 0 < 5 max                     │
│ ✓ Total exposure: $56,406 (11.3%) ≤ 50% limit  │
└─────────────────────────────────────────────────┘

CHECK 3: Correlation Check (lines 385-444)
┌─────────────────────────────────────────────────┐
│ ✓ No existing positions to correlate with       │
│ ✓ No size reduction needed                      │
└─────────────────────────────────────────────────┘

CHECK 4: Position Sizing (backend/execution/position_sizer.py)
┌─────────────────────────────────────────────────┐
│ Risk Amount: $100,000 × 1% = $1,000            │
│ Price Risk: $451.25 - $443.25 = $8.00          │
│ Shares: $1,000 / $8.00 = 125 shares            │
│ ✓ Within deployment limit (50%)                 │
│ ✓ Kill-switch not active                        │
└─────────────────────────────────────────────────┘

CHECK 5: Circuit Breaker (backend/risk/kill_switch.py)
┌─────────────────────────────────────────────────┐
│ Current Drawdown: 0%                            │
│ Status: NORMAL (< 15%)                          │
│ ✓ Trading ACTIVE                                │
└─────────────────────────────────────────────────┘

ALL CHECKS PASSED → Proceed to order
```

### Step 7: Order Placement

```
File: backend/execution/order_manager.py (lines 158-204)

async def place_order_with_monitoring(signal: SignalExplanation):
    order = await self.broker.place_order(
        symbol="SPY",
        qty=125,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    # Track order for monitoring
    self.managed_orders[order.order_id] = ManagedOrder(
        order=order,
        placed_at=datetime.now(),
        chase_count=0
    )

    return order

Alpaca API Call:
POST https://paper-api.alpaca.markets/v2/orders
{
    "symbol": "SPY",
    "qty": 125,
    "side": "buy",
    "type": "market",
    "time_in_force": "day"
}
```

### Step 8: Order Fill

```
File: backend/execution/broker_interface.py (lines 367-417)

Alpaca Response:
{
    "id": "ord_abc123",
    "status": "filled",
    "filled_qty": 125,
    "filled_avg_price": 451.28,
    "filled_at": "2024-01-22T14:30:01Z"
}

Position Created:
Position(
    symbol="SPY",
    qty=125,
    side=1,  # LONG
    entry_price=451.28,
    current_price=451.28,
    unrealized_pnl=0.00,
    stop_loss=443.25,
    entry_time="2024-01-22T14:30:01Z"
)
```

### Step 9: Persistence

```
File: backend/storage/sqlite_store.py

Trade Record:
INSERT INTO trades (
    symbol, side, entry_price, entry_time, shares,
    stop_loss, status, strategy, regime, signal_reason
) VALUES (
    'SPY', 1, 451.28, '2024-01-22T14:30:01Z', 125,
    443.25, 'OPEN', 'TREND', 'STRONG_TREND',
    'BUY SPY @ $451.25 | KAMA crossover | Risk: $1,000'
);

Position Record:
INSERT INTO positions (
    symbol, side, shares, entry_price, current_price,
    unrealized_pnl, stop_loss, entry_time
) VALUES (
    'SPY', 1, 125, 451.28, 451.28, 0.00, 443.25,
    '2024-01-22T14:30:01Z'
);
```

### Step 10: Frontend Update

```
WebSocket to Frontend:
{
    "type": "position_update",
    "position": {
        "symbol": "SPY",
        "side": "LONG",
        "shares": 125,
        "entry_price": 451.28,
        "current_price": 451.28,
        "unrealized_pnl": 0.00,
        "stop_loss": 443.25
    }
}

GET /api/positions Response:
[{
    "symbol": "SPY",
    "side": 1,
    "shares": 125,
    "entry_price": 451.28,
    "current_price": 451.28,
    "unrealized_pnl": 0.00,
    "pnl_percent": 0.00,
    "stop_loss": 443.25,
    "entry_time": "2024-01-22T14:30:01Z"
}]
```

### Final State

```
Account:
- Equity: $100,000
- Cash: $43,590 ($100,000 - $56,410)
- Positions Value: $56,410

Position:
- SPY: 125 shares @ $451.28
- Stop Loss: $443.25 (risk: $1,003.75)
- Unrealized P&L: $0.00
```

---

## Scenario 3: Mean Reversion Short Entry

**Context**: AAPL is in a ranging market. RSI hits overbought and price touches upper Bollinger Band.

### Market Conditions

```
AAPL Current State:
- Price: $185.50
- RSI (14): 74 (Overbought: > 70)
- ADX (14): 18 (Weak trend: < 20)
- R² (20): 0.32 (Mean-reverting: < 0.4)
- Upper BB: $185.20
- 20-SMA: $182.00
- Lower BB: $178.80
```

### Regime Detection

```
File: backend/strategy/regime_detector.py

def classify_trend_regime(adx=18, r_squared=0.32):
    # ADX = 18 < 20 ✓
    # R² = 0.32 < 0.4 ✓
    return REGIME_MEAN_REVERSION  # Code: 0
```

### Signal Generation (Mean Reversion Mode)

```
File: backend/strategy/dual_mode.py (lines 181-260)

Entry Conditions:
1. RSI > 70 ✓ (74 > 70)
2. Price at Upper BB ✓ ($185.50 > $185.20)
→ SHORT signal generated

def generate_mean_reversion_signals(price, rsi, bb_upper, bb_lower, sma):
    if rsi > 70 and price >= bb_upper:
        return SignalType.SHORT

Stop Loss: Fixed 3% above entry
- Stop = $185.50 × 1.03 = $191.07

Take Profit: Return to 20-SMA
- Target = $182.00
```

### Risk Calculation

```
Mean Reversion Risk: 0.75% (vs 1% for trend)
Risk Amount: $100,000 × 0.0075 = $750
Price Risk: $191.07 - $185.50 = $5.57
Shares: $750 / $5.57 = 134 shares
Position Value: 134 × $185.50 = $24,857 (24.9%)
```

### Signal Explanation

```
SignalExplanation:
    signal_type: SHORT (-1)
    symbol: "AAPL"
    price: 185.50

    strategy_mode: "MEAN_REVERSION"
    regime: "MEAN_REVERSION"

    rsi: 74
    adx: 18
    r_squared: 0.32

    risk_amount: $750
    risk_percent: 0.0075
    stop_loss: 191.07
    take_profit: 182.00
    position_size: 134 shares

    entry_trigger: "RSI overbought + Upper BB touch"

Formatted:
"SHORT AAPL @ $185.50
 Reason: RSI=74 (overbought) + Price at Upper BB
 Regime: MEAN_REVERSION (ADX=18, R²=0.32)
 Risk: $750 (0.75% account), Stop: $191.07, Target: $182.00"
```

### Order Execution

```
POST /v2/orders
{
    "symbol": "AAPL",
    "qty": 134,
    "side": "sell",     // Short sale
    "type": "market",
    "time_in_force": "day"
}

Fill: 134 shares @ $185.48
```

### Final State (Two Positions)

```
Account:
- Equity: $100,000
- Cash: $68,464
- Positions Value: $81,255

Positions:
1. SPY LONG: 125 shares @ $451.28, P&L: $0
2. AAPL SHORT: 134 shares @ $185.48, P&L: $0

Total Exposure: 81.3% (approaching 50% soft limit)
```

---

## Scenario 4: Stop Loss Hit

**Context**: SPY drops and hits the trailing stop loss.

### Price Movement

```
Time Series:
14:30 - Entry @ $451.28, Stop @ $443.25
14:45 - Price $452.00, Trail stop to $444.00
15:00 - Price $450.50, Stop remains $444.00
15:15 - Price $448.00, Stop remains $444.00
15:30 - Price $443.50, STOP HIT!
```

### Stop Check on Each Price Update

```
File: backend/execution/broker_interface.py (lines 247-265)

def set_market_price(self, symbol: str, price: float):
    self.market_prices[symbol] = price

    if symbol in self.positions:
        position = self.positions[symbol]
        position.current_price = price
        position.unrealized_pnl = (price - position.entry_price) * position.qty

        # Check stop loss
        if position.side == 1:  # LONG
            if price <= position.stop_loss:
                self._trigger_stop_loss(position)

At 15:30:
- Price: $443.50
- Stop: $444.00
- $443.50 < $444.00 → STOP TRIGGERED
```

### Stop Loss Order

```
File: backend/execution/order_manager.py

def _trigger_stop_loss(self, position: Position):
    # Place market sell order
    order = await self.broker.place_order(
        symbol=position.symbol,
        qty=position.qty,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET
    )

POST /v2/orders
{
    "symbol": "SPY",
    "qty": 125,
    "side": "sell",
    "type": "market"
}

Fill: 125 shares @ $443.45
```

### Trade Closed

```
Trade Update:
UPDATE trades SET
    exit_price = 443.45,
    exit_time = '2024-01-22T15:30:05Z',
    realized_pnl = -978.75,
    status = 'CLOSED'
WHERE symbol = 'SPY' AND status = 'OPEN';

Calculation:
Entry: $451.28 × 125 = $56,410.00
Exit: $443.45 × 125 = $55,431.25
P&L: -$978.75 (-1.74%)
Commission: ~$1.25 (125 × $0.005 × 2)
Net P&L: -$980.00
```

### Account Update

```
Before Stop:
- Equity: $100,000
- SPY Position: $56,410

After Stop:
- Equity: $99,021.25 (-0.98% drawdown)
- Cash: $99,021.25 (position closed)
- SPY Position: CLOSED

Drawdown Tracker Update:
- Peak Equity: $100,000
- Current Equity: $99,021.25
- Drawdown: 0.98% (NORMAL, < 15%)
```

---

## Scenario 5: Circuit Breaker Activation

**Context**: Multiple losing trades push drawdown to 15%, then 20%.

### Drawdown Progression

```
Starting Equity: $100,000

Trade 1: SPY -$980 → Equity: $99,020 (DD: 0.98%)
Trade 2: AAPL -$1,200 → Equity: $97,820 (DD: 2.18%)
Trade 3: MSFT -$1,500 → Equity: $96,320 (DD: 3.68%)
...
Trade 10: QQQ -$2,100 → Equity: $85,500 (DD: 14.5%)
Trade 11: NVDA -$800 → Equity: $84,700 (DD: 15.3%)
```

### WARNING Level Triggered (15% Drawdown)

```
File: backend/risk/kill_switch.py (lines 254-343)

def check_drawdown_level(drawdown_pct: float) -> DrawdownLevel:
    if drawdown_pct >= 0.20:
        return DrawdownLevel.CRITICAL
    elif drawdown_pct >= 0.15:
        return DrawdownLevel.WARNING  # ← 15.3% hits this
    return DrawdownLevel.NORMAL

Actions at WARNING (lines 276-343):
1. TradingStatus → REDUCED
2. Position size multiplier → 0.5 (50% reduction)
3. Stop loss multiplier → 2.0× ATR (from 2.5×, tighter)
4. Alert generated

def get_position_size_multiplier(status: TradingStatus) -> float:
    if status == TradingStatus.REDUCED:
        return 0.5  # Half size
    return 1.0

Alert Message:
"WARNING: Drawdown at 15.3%. Position sizes reduced 50%.
 Stops tightened to 2.0× ATR. Review recommended."
```

### New Trade with Reduced Size

```
Normal calculation: 125 shares
With WARNING status: 125 × 0.5 = 62 shares

Signal generated but size halved:
- Original risk: $1,000 (1%)
- Reduced risk: $500 (0.5%)
- Original shares: 125
- Reduced shares: 62
```

### CRITICAL Level Triggered (20% Drawdown)

```
Trade 12: AMZN -$1,800 → Equity: $82,900 (DD: 17.1%)
Trade 13: GOOGL -$2,500 → Equity: $80,400 (DD: 19.6%)
Trade 14: META -$600 → Equity: $79,800 (DD: 20.2%)

def check_drawdown_level(drawdown_pct=0.202):
    return DrawdownLevel.CRITICAL  # 20.2% >= 20%

Actions at CRITICAL (lines 319-343):
1. TradingStatus → DISABLED
2. Close ALL open positions immediately
3. Block all new orders
4. Require manual review to resume
```

### Emergency Position Close

```
File: backend/risk/kill_switch.py

async def emergency_close_all_positions():
    positions = await self.broker.get_positions()
    for position in positions:
        await self.broker.place_order(
            symbol=position.symbol,
            qty=position.qty,
            side=OrderSide.SELL if position.side == 1 else OrderSide.BUY,
            order_type=OrderType.MARKET
        )

    # Log emergency action
    logger.critical(
        "CIRCUIT BREAKER: All positions closed. "
        f"Drawdown: 20.2%. Trading DISABLED."
    )
```

### Trading Blocked

```
New signal arrives for TSLA BUY...

File: backend/risk/kill_switch.py (lines 389-411)

def can_open_new_position(self) -> tuple[bool, str]:
    if self.trading_status == TradingStatus.DISABLED:
        return (False, "Trading disabled: 20% drawdown breached")
    return (True, "")

Result: Order REJECTED
Reason: "Trading disabled: 20% drawdown breached. Manual review required."

To Resume:
1. Admin reviews losing trades
2. Identifies issues (strategy, market conditions)
3. Manually resets circuit breaker
4. Trading resumes with REDUCED status
```

---

## Scenario 6: Order Chase Logic

**Context**: Limit order doesn't fill immediately. System chases the price.

### Initial Limit Order

```
Signal: BUY SPY @ $450.00 (limit)
Current Price: $450.25

POST /v2/orders
{
    "symbol": "SPY",
    "qty": 100,
    "side": "buy",
    "type": "limit",
    "limit_price": "450.00",
    "time_in_force": "day"
}

Order Status: PENDING
```

### Order Monitoring Loop

```
File: backend/execution/order_manager.py (lines 206-219)

async def _monitoring_loop(self):
    while self._running:
        await asyncio.sleep(5)  # Poll every 5 seconds
        await self._check_all_orders()

Timeline:
T+0s:  Order placed, status=PENDING
T+5s:  Check: PENDING, price=$450.30
T+10s: Check: PENDING, price=$450.35
T+15s: Check: PENDING, price=$450.40
...
T+60s: Check: PENDING, price=$450.50 → CHASE TRIGGERED
```

### Chase Logic (After 60 Seconds)

```
File: backend/execution/order_manager.py (lines 268-340)

async def _chase_order(self, managed_order: ManagedOrder):
    # Check if order should be chased
    elapsed = (datetime.now() - managed_order.placed_at).total_seconds()
    if elapsed < 60:
        return

    if managed_order.chase_count >= 3:
        # Max chases reached, abandon
        await self.broker.cancel_order(managed_order.order.order_id)
        logger.warning(f"Order abandoned after 3 chases: {managed_order.order}")
        return

    # Cancel existing order
    await self.broker.cancel_order(managed_order.order.order_id)

    # Recalculate mid-price
    current_price = await self.broker.get_current_price("SPY")
    new_limit = current_price - 0.02  # Slightly below market

    # Place new order
    new_order = await self.broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=new_limit  # $450.48
    )

    # Update tracking
    managed_order.order = new_order
    managed_order.placed_at = datetime.now()
    managed_order.chase_count += 1
```

### Chase Timeline

```
Chase 1 (T+60s):
- Cancel order @ $450.00
- New order @ $450.48
- Status: PENDING

Chase 2 (T+120s):
- Price moved to $450.80
- Cancel order @ $450.48
- New order @ $450.78
- Status: PENDING

Chase 3 (T+180s):
- Price moved to $451.00
- Cancel order @ $450.78
- New order @ $450.98
- Status: FILLED @ $450.98

Total slippage: $0.98 from original $450.00 target
```

### Max Chase Reached (Alternative)

```
If after Chase 3 still not filled:

T+240s: Check order
- chase_count = 3 (max)
- Order ABANDONED

logger.warning(
    "Order abandoned: SPY BUY 100 @ limit. "
    "3 chases failed. Market moved significantly."
)

Action: No position opened, signal expired
```

---

## Scenario 7: Correlation-Based Size Reduction

**Context**: Already holding SPY long, new signal to buy QQQ (highly correlated).

### Existing Position

```
Current Holdings:
- SPY LONG: 125 shares @ $451.28
- Correlation(SPY, QQQ) = 0.85 (high!)
```

### New Signal

```
QQQ BUY signal generated:
- Entry: $380.00
- Stop: $372.00
- Normal size: 130 shares (based on 1% risk)
```

### Correlation Check

```
File: backend/risk/position_limits.py (lines 385-444)

def check_correlation_with_existing_positions(
    new_symbol: str,
    new_symbol_prices: np.ndarray,
    existing_positions: List[Position],
    position_prices_map: Dict[str, np.ndarray],
    threshold: float = 0.7
) -> Tuple[bool, float, str]:

    for position in existing_positions:
        if position.symbol == "SPY":
            spy_prices = position_prices_map["SPY"]

            # Calculate correlation
            correlation = np.corrcoef(
                new_symbol_prices,  # QQQ
                spy_prices          # SPY
            )[0, 1]

            if abs(correlation) > threshold:
                return (True, correlation, position.symbol)

    return (False, 0.0, "")

Result:
- should_reduce = True
- correlation = 0.85
- correlated_with = "SPY"
```

### Size Reduction Applied

```
File: backend/risk/position_limits.py (lines 452-534)

def validate_new_position(...) -> Tuple[RiskCheckResult, str, int]:
    # Run correlation check
    should_reduce, corr, corr_symbol = check_correlation_with_existing_positions(...)

    if should_reduce:
        # Apply 50% reduction
        adjusted_shares = int(original_shares * 0.5)
        reason = f"Size reduced 50%: {new_symbol} correlated {corr:.2f} with {corr_symbol}"

    return (RiskCheckResult.APPROVED_WITH_ADJUSTMENT, reason, adjusted_shares)

Calculation:
- Original shares: 130
- Correlation: 0.85 (> 0.7 threshold)
- Reduction: 50%
- Adjusted shares: 65

Warning logged:
"QQQ size reduced from 130 to 65 shares.
 Reason: Correlation 0.85 with existing SPY position."
```

### Final Order

```
POST /v2/orders
{
    "symbol": "QQQ",
    "qty": 65,        // Reduced from 130
    "side": "buy",
    "type": "market"
}

Portfolio After:
- SPY LONG: 125 shares @ $451.28 (value: $56,410)
- QQQ LONG: 65 shares @ $380.05 (value: $24,703)
- Total Exposure: $81,113 (81.1%)
- Correlation Risk: Acknowledged and mitigated
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ALPACA MARKET DATA                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   REST API                              WebSocket                             │
│   GET /v2/stocks/{symbol}/bars          wss://stream.data.alpaca.markets     │
│   ─────────────────────────             ────────────────────────────────     │
│   • Historical candles (500)            • Real-time 1-min bars               │
│   • Rate: 200 req/min                   • Heartbeat: 60s timeout             │
│   • Retry: 3× with backoff              • Auto-reconnect: 5 retries          │
│                                                                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                       │
│                   backend/data/fetcher.py                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   DataPipeline.start()                  DataPipeline.process_stream()        │
│   ─────────────────────                 ────────────────────────────         │
│   1. fetch_candles(500)                 • Continuous WebSocket listen        │
│   2. ws_feed.connect()                  • Parse incoming bars                │
│   3. ws_feed.subscribe()                • Trigger signal_callback            │
│   4. Return initial data                • Non-blocking (asyncio.create_task) │
│                                                                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        TECHNICAL ANALYSIS                                     │
│                   backend/computation/indicators.py                           │
│                   backend/strategy/regime_detector.py                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   Indicators (Numba JIT)                Regime Detection                      │
│   ─────────────────────                 ────────────────                     │
│   • EMA, SMA, KAMA                      • ADX calculation                    │
│   • RSI (0-100)                         • R² linear regression               │
│   • ATR (volatility)                    • Classify: TREND/MR/NEUTRAL         │
│   • Bollinger Bands                     • 3-bar persistence filter           │
│                                                                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        SIGNAL GENERATION                                      │
│                   backend/strategy/dual_mode.py                               │
│                   backend/strategy/signal_generator.py                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   Dual-Mode Strategy                    SignalExplanation                    │
│   ─────────────────                     ─────────────────                    │
│   TREND (ADX>25, R²>0.6):               • signal_type (LONG/SHORT/EXIT)      │
│   • KAMA crossover entry                • strategy_mode, regime              │
│   • Trailing 2.5×ATR stop               • All indicator values               │
│                                         • risk_amount, stop_loss             │
│   MEAN_REV (ADX<20, R²<0.4):            • position_size                      │
│   • RSI + BB entry                      • entry_trigger explanation          │
│   • Fixed 3% stop                       • to_dict() for storage              │
│                                                                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         RISK VALIDATION                                       │
│                   backend/risk/position_limits.py                             │
│                   backend/risk/kill_switch.py                                 │
│                   backend/execution/position_sizer.py                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   5-Point Risk Check:                                                        │
│   ───────────────────                                                        │
│   1. Position-Level: stop required, max 1% risk, max 20% size               │
│   2. Portfolio-Level: max 5 positions, max 50% exposure                     │
│   3. Correlation: if >0.7, reduce size 50%                                  │
│   4. Position Sizer: 1% rule, deployment limit, round down                  │
│   5. Circuit Breaker: NORMAL/WARNING(15%)/CRITICAL(20%)                     │
│                                                                               │
│   Results: APPROVED | APPROVED_WITH_ADJUSTMENT | REJECTED                    │
│                                                                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ORDER MANAGEMENT                                       │
│                   backend/execution/order_manager.py                          │
│                   backend/execution/broker_interface.py                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   OrderManager                          BrokerInterface                      │
│   ────────────                          ───────────────                      │
│   • place_order_with_monitoring()       • place_order() → Alpaca            │
│   • 5-second polling loop               • cancel_order()                     │
│   • 60-second chase trigger             • get_order_status()                 │
│   • Max 3 chases then abandon           • get_positions()                    │
│   • Track ManagedOrder state            • get_account()                      │
│                                                                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ALPACA ORDERS                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   POST /v2/orders                       Order Lifecycle                      │
│   ───────────────                       ───────────────                      │
│   {                                     PENDING → FILLED                     │
│     "symbol": "SPY",                           ↓                             │
│     "qty": 125,                         Position Created                     │
│     "side": "buy",                             ↓                             │
│     "type": "market"                    P&L Tracking                         │
│   }                                            ↓                             │
│                                         Stop/Exit Check                      │
│                                                                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          PERSISTENCE                                          │
│                   backend/storage/sqlite_store.py                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   Async Write Queue                     Tables                               │
│   ────────────────                      ──────                               │
│   • Non-blocking writes                 • trades (history)                   │
│   • <5ms latency                        • positions (current)                │
│   • 30-day retention                    • settings (config)                  │
│                                                                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER                                           │
│                   backend/api/server.py                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   REST Endpoints                        WebSocket                            │
│   ──────────────                        ─────────                            │
│   GET /api/positions                    /ws/prices                           │
│   GET /api/trades                       • Price updates (2s)                 │
│   GET /api/account                      • Position updates                   │
│   GET /api/status                       • Auth required                      │
│   POST /api/backtest                                                         │
│                                                                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND                                            │
│                   React/TypeScript Application                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   • WebSocket client for live prices                                         │
│   • REST polling for positions/account                                       │
│   • Real-time P&L display                                                    │
│   • Trade history table                                                      │
│   • System status indicators                                                 │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Code References

| Scenario | File | Lines | Function |
|----------|------|-------|----------|
| Data Fetch | `backend/data/fetcher.py` | 291-351 | `fetch_candles()` |
| WebSocket | `backend/data/fetcher.py` | 582-633 | `stream()` |
| Regime Detection | `backend/strategy/regime_detector.py` | 362-421 | `classify_trend_regime()` |
| Trend Signal | `backend/strategy/dual_mode.py` | 35-123 | `generate_trend_following_signals()` |
| Mean Rev Signal | `backend/strategy/dual_mode.py` | 181-260 | `generate_mean_reversion_signals()` |
| Signal Explanation | `backend/strategy/signal_generator.py` | 298-393 | `generate_signal_with_explanation()` |
| Position Risk | `backend/risk/position_limits.py` | 160-226 | `validate_position_level_risk()` |
| Portfolio Risk | `backend/risk/position_limits.py` | 277-326 | `validate_portfolio_level_risk()` |
| Correlation Check | `backend/risk/position_limits.py` | 385-444 | `check_correlation_with_existing_positions()` |
| Position Sizing | `backend/execution/position_sizer.py` | 108-265 | `calculate_position_size()` |
| Circuit Breaker | `backend/risk/kill_switch.py` | 254-343 | `check_drawdown_level()` |
| Order Placement | `backend/execution/order_manager.py` | 158-204 | `place_order_with_monitoring()` |
| Order Chase | `backend/execution/order_manager.py` | 268-340 | `_chase_order()` |
| Stop Loss | `backend/execution/broker_interface.py` | 247-265 | `set_market_price()` |
| Persistence | `backend/storage/sqlite_store.py` | 145-200 | `add_trade()`, `add_position()` |

---

*Document created for FluxHero Trading System v1.0.0*
