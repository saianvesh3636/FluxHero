#!/usr/bin/env python3
"""
Test Data Seeding Script for FluxHero

Creates sample positions with realistic data for development and testing.

Requirements (from TASKS.md Phase 5, Task 5.2):
- Create 5-10 sample positions with realistic data
- Positions have realistic P&L values
- Make seed-data command available via Makefile

Usage:
    python scripts/seed_test_data.py [--count N]
    make seed-data
"""

import asyncio
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.execution.broker_interface import OrderSide, OrderType, PaperBroker
from backend.storage.sqlite_store import SQLiteStore

# Sample symbols with typical characteristics
SYMBOLS = [
    {"symbol": "SPY", "price_range": (400, 450), "volatility": 0.01},  # Low vol ETF
    {"symbol": "QQQ", "price_range": (350, 400), "volatility": 0.015},  # Tech ETF
    {"symbol": "AAPL", "price_range": (170, 185), "volatility": 0.02},  # Blue chip
    {"symbol": "TSLA", "price_range": (200, 280), "volatility": 0.04},  # High vol
    {"symbol": "NVDA", "price_range": (400, 500), "volatility": 0.035},  # High growth
    {"symbol": "MSFT", "price_range": (370, 400), "volatility": 0.018},  # Blue chip
    {"symbol": "AMZN", "price_range": (145, 160), "volatility": 0.025},  # Tech
    {"symbol": "META", "price_range": (400, 450), "volatility": 0.03},  # Tech
    {"symbol": "GOOGL", "price_range": (135, 145), "volatility": 0.02},  # Tech
    {"symbol": "AMD", "price_range": (130, 160), "volatility": 0.035},  # Semiconductor
]


def generate_realistic_position_data() -> dict:
    """
    Generate realistic position data with proper P&L characteristics.

    Returns:
        dict: Position parameters including symbol, entry price, current price, etc.
    """
    # Select random symbol
    symbol_data = random.choice(SYMBOLS)
    symbol = symbol_data["symbol"]
    price_min, price_max = symbol_data["price_range"]
    # volatility could be used for more sophisticated P&L modeling in the future

    # Generate entry price within range
    entry_price = random.uniform(price_min, price_max)

    # Determine if position is winning or losing (60% win rate for realism)
    is_winning = random.random() < 0.6

    # Generate current price based on entry price and volatility
    # Winning positions: 0.5% to 5% profit
    # Losing positions: -0.3% to -3% loss
    if is_winning:
        pnl_pct = random.uniform(0.005, 0.05)  # 0.5% to 5% profit
    else:
        pnl_pct = random.uniform(-0.03, -0.003)  # -3% to -0.3% loss

    current_price = entry_price * (1 + pnl_pct)

    # Determine position side (80% long, 20% short for realism)
    side = OrderSide.BUY if random.random() < 0.8 else OrderSide.SELL

    # Calculate position size based on account risk (1% risk per trade)
    # Assuming $100k account, max position size would be around 200-500 shares
    # depending on stock price
    account_value = 100000
    max_position_value = account_value * 0.2  # 20% max per position
    max_shares = int(max_position_value / entry_price)
    shares = random.randint(max(10, max_shares // 4), max_shares)

    # Generate entry time (within last 1-30 days)
    days_ago = random.randint(1, 30)
    hours_ago = random.randint(0, 23)
    entry_time = datetime.now() - timedelta(days=days_ago, hours=hours_ago)

    # Determine strategy and regime based on position characteristics
    if is_winning and abs(pnl_pct) > 0.02:
        strategy = "TREND"
        regime = "STRONG_TREND"
    elif not is_winning:
        strategy = "MEAN_REVERSION"
        regime = "MEAN_REVERSION"
    else:
        strategy = random.choice(["TREND", "MEAN_REVERSION"])
        regime = "NEUTRAL"

    # Calculate stop loss (2.5% for trend, 3% for mean reversion)
    stop_distance_pct = 0.025 if strategy == "TREND" else 0.03
    if side == OrderSide.BUY:
        stop_loss = entry_price * (1 - stop_distance_pct)
    else:
        stop_loss = entry_price * (1 + stop_distance_pct)

    # Generate signal reason
    signal_reasons = {
        "TREND": [
            f"Price crossed above KAMA + 0.5×ATR (Entry: ${entry_price:.2f})",
            f"Strong uptrend detected (ADX>30, R²=0.{random.randint(75, 95)})",
            f"Breakout confirmed with volume spike ({random.uniform(1.5, 3.0):.1f}× avg)",
        ],
        "MEAN_REVERSION": [
            f"RSI oversold ({random.randint(15, 30)}) at lower Bollinger Band",
            f"Price touched support with low volatility (ATR={random.uniform(2, 5):.2f})",
            f"Mean reversion signal in ranging market (ADX={random.randint(12, 20)})",
        ],
    }
    signal_reason = random.choice(signal_reasons[strategy])

    return {
        "symbol": symbol,
        "entry_price": round(entry_price, 2),
        "current_price": round(current_price, 2),
        "shares": shares,
        "side": side,
        "stop_loss": round(stop_loss, 2),
        "entry_time": entry_time,
        "strategy": strategy,
        "regime": regime,
        "signal_reason": signal_reason,
    }


async def seed_positions(count: int = 7) -> None:
    """
    Seed the database with sample positions.

    Args:
        count: Number of positions to create (default: 7, range: 5-10)
    """
    print(f"🌱 Seeding {count} test positions...")

    # Initialize storage
    store = SQLiteStore()
    await store.initialize()

    # Initialize paper broker
    broker = PaperBroker(initial_capital=100000.0)

    # Track created positions for summary
    created_positions = []

    try:
        # Generate and place orders for each position
        for i in range(count):
            position_data = generate_realistic_position_data()

            # Set market price before placing order
            broker.set_market_price(position_data["symbol"], position_data["entry_price"])

            # Place order via paper broker
            _ = await broker.place_order(
                symbol=position_data["symbol"],
                qty=position_data["shares"],
                side=position_data["side"],
                order_type=OrderType.MARKET,
            )

            # Update broker's market prices to reflect current P&L
            broker.set_market_price(position_data["symbol"], position_data["current_price"])

            # Calculate P&L
            if position_data["side"] == OrderSide.BUY:
                unrealized_pnl = (
                    (position_data["current_price"] - position_data["entry_price"])
                    * position_data["shares"]
                )
            else:
                unrealized_pnl = (
                    (position_data["entry_price"] - position_data["current_price"])
                    * position_data["shares"]
                )

            position_value = position_data["entry_price"] * position_data["shares"]
            pnl_pct = (unrealized_pnl / position_value) * 100

            created_positions.append({
                "symbol": position_data["symbol"],
                "side": "LONG" if position_data["side"] == OrderSide.BUY else "SHORT",
                "shares": position_data["shares"],
                "entry_price": position_data["entry_price"],
                "current_price": position_data["current_price"],
                "unrealized_pnl": unrealized_pnl,
                "pnl_pct": pnl_pct,
                "strategy": position_data["strategy"],
                "regime": position_data["regime"],
            })

            print(f"  ✓ {i+1}/{count}: {position_data['symbol']} "
                  f"{position_data['shares']} shares @ ${position_data['entry_price']:.2f} "
                  f"({position_data['strategy']}, P&L: ${unrealized_pnl:+.2f} / {pnl_pct:+.2f}%)")

        print(f"\n✅ Successfully seeded {count} positions")
        print("\n📊 Position Summary:")
        print("-" * 90)
        header = (
            f"{'Symbol':<8} {'Side':<6} {'Shares':<8} {'Entry':<10} "
            f"{'Current':<10} {'P&L $':<12} {'P&L %':<10} {'Strategy':<15}"
        )
        print(header)
        print("-" * 90)

        total_pnl = 0
        for pos in created_positions:
            print(f"{pos['symbol']:<8} {pos['side']:<6} {pos['shares']:<8} "
                  f"${pos['entry_price']:<9.2f} ${pos['current_price']:<9.2f} "
                  f"${pos['unrealized_pnl']:+11.2f} {pos['pnl_pct']:+9.2f}% "
                  f"{pos['strategy']:<15}")
            total_pnl += pos['unrealized_pnl']

        print("-" * 90)
        print(f"{'TOTAL':<53} ${total_pnl:+11.2f}")
        print("-" * 90)

        # Display account info
        account = await broker.get_account()
        print("\n💰 Account Summary:")
        print(f"  Balance: ${account.balance:,.2f}")
        print(f"  Equity: ${account.equity:,.2f}")
        print(f"  Cash: ${account.cash:,.2f}")
        print(f"  Positions Value: ${account.positions_value:,.2f}")

    finally:
        await store.close()


async def clear_positions() -> None:
    """Clear all existing test positions."""
    print("🧹 Clearing existing positions...")

    store = SQLiteStore()
    await store.initialize()

    try:
        # Get all positions
        positions = await store.get_open_positions()

        # Delete each position
        for position in positions:
            await store.delete_position(position.symbol)

        print(f"  ✓ Cleared {len(positions)} positions")
    finally:
        await store.close()


def main():
    """Main entry point for the seeding script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Seed FluxHero database with test positions"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=7,
        help="Number of positions to create (5-10, default: 7)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing positions before seeding"
    )

    args = parser.parse_args()

    # Validate count
    if args.count < 5 or args.count > 10:
        print("⚠️  Warning: Count should be between 5 and 10. Adjusting...")
        args.count = max(5, min(10, args.count))

    # Run seeding
    try:
        if args.clear:
            asyncio.run(clear_positions())

        asyncio.run(seed_positions(args.count))

        print("\n🎉 Seeding complete! You can now:")
        print("  1. Start the backend: make dev-backend")
        print("  2. View positions at: http://localhost:8000/api/positions")
        print("  3. Open frontend: http://localhost:3000/live")

    except KeyboardInterrupt:
        print("\n\n⚠️  Seeding interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during seeding: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
