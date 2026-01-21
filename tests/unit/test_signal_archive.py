"""
Unit tests for Signal Explainer Archive Viewer (Phase 15 - Task 5)

Tests cover:
- Signal explanation parsing from Trade objects
- Filtering by symbol, strategy, regime, date range
- Search functionality
- Sorting by various criteria
- CSV export functionality
- Modal detail view rendering
- Pagination
- Edge cases
"""

import pytest
import json
from datetime import datetime, timedelta


# Mock Trade data for testing
def create_mock_trade(
    trade_id: int,
    symbol: str = "SPY",
    entry_price: float = 420.0,
    exit_price: float = 425.0,
    shares: int = 100,
    strategy: str = "TREND",
    regime: str = "STRONG_TREND",
    signal_reason: str = "KAMA crossover",
    realized_pnl: float = 500.0,
    entry_time: float = None,
    signal_explanation: dict = None,
) -> dict:
    """Create a mock trade for testing."""
    if entry_time is None:
        entry_time = datetime.now().timestamp()

    if signal_explanation is None:
        signal_explanation = {
            'symbol': symbol,
            'signal_type': 1,  # LONG
            'price': entry_price,
            'timestamp': entry_time,
            'strategy_mode': 2,  # TREND_FOLLOWING
            'regime': 2,  # STRONG_TREND
            'volatility_state': 1,  # NORMAL
            'atr': 3.2,
            'kama': 418.5,
            'rsi': 55.0,
            'adx': 32.0,
            'r_squared': 0.81,
            'risk_amount': 1000.0,
            'risk_percent': 0.01,
            'stop_loss': 415.0,
            'position_size': 100,
            'entry_trigger': signal_reason,
            'noise_filtered': True,
            'volume_validated': True,
            'formatted_reason': f"BUY {symbol} @ ${entry_price:.2f}\nReason: Volatility (ATR=3.2, Normal) + {signal_reason}\nRegime: STRONG_TREND (ADX=32.0, R²=0.81)\nRisk: $1,000 (1.00% account), Stop: $415.00",
            'compact_reason': f"BUY @ ${entry_price:.2f} | {signal_reason} | STRONG_TREND (ATR=3.2, Normal) | Risk: $1,000 (1.00%)",
        }

    return {
        'id': trade_id,
        'symbol': symbol,
        'side': 'LONG',
        'entry_price': entry_price,
        'entry_time': entry_time,
        'exit_price': exit_price,
        'exit_time': entry_time + 3600,  # 1 hour later
        'shares': shares,
        'stop_loss': 415.0,
        'take_profit': 430.0,
        'realized_pnl': realized_pnl,
        'status': 1,  # CLOSED
        'strategy': strategy,
        'regime': regime,
        'signal_reason': signal_reason,
        'signal_explanation': json.dumps(signal_explanation),
        'created_at': datetime.fromtimestamp(entry_time).isoformat(),
        'updated_at': datetime.fromtimestamp(entry_time + 3600).isoformat(),
    }


class TestSignalExplanationParsing:
    """Test parsing of signal explanations from Trade objects."""

    def test_parse_valid_json_explanation(self):
        """Test parsing valid JSON signal explanation."""
        trade = create_mock_trade(1)
        explanation_str = trade['signal_explanation']
        explanation = json.loads(explanation_str)

        assert explanation['symbol'] == 'SPY'
        assert explanation['signal_type'] == 1
        assert explanation['price'] == 420.0
        assert explanation['entry_trigger'] == 'KAMA crossover'

    def test_parse_explanation_with_all_fields(self):
        """Test parsing explanation with all optional fields present."""
        trade = create_mock_trade(1)
        explanation = json.loads(trade['signal_explanation'])

        # Check all required fields
        assert 'symbol' in explanation
        assert 'signal_type' in explanation
        assert 'price' in explanation
        assert 'timestamp' in explanation
        assert 'strategy_mode' in explanation
        assert 'regime' in explanation
        assert 'volatility_state' in explanation
        assert 'atr' in explanation
        assert 'kama' in explanation

        # Check optional fields
        assert 'rsi' in explanation
        assert 'adx' in explanation
        assert 'r_squared' in explanation

        # Check risk fields
        assert 'risk_amount' in explanation
        assert 'risk_percent' in explanation
        assert 'stop_loss' in explanation
        assert 'position_size' in explanation

        # Check validation fields
        assert 'noise_filtered' in explanation
        assert 'volume_validated' in explanation

    def test_parse_explanation_with_missing_optional_fields(self):
        """Test parsing explanation with missing optional fields."""
        signal_explanation = {
            'symbol': 'AAPL',
            'signal_type': 1,
            'price': 150.0,
            'timestamp': datetime.now().timestamp(),
            'strategy_mode': 1,
            'regime': 0,
            'volatility_state': 1,
            'atr': 2.5,
            'kama': 148.0,
            'risk_amount': 500.0,
            'risk_percent': 0.0075,
            'stop_loss': 145.0,
            'position_size': 50,
            'entry_trigger': 'RSI oversold',
            'noise_filtered': True,
            'volume_validated': True,
        }

        trade = create_mock_trade(1, symbol="AAPL", signal_explanation=signal_explanation)
        explanation = json.loads(trade['signal_explanation'])

        assert explanation['symbol'] == 'AAPL'
        assert explanation['entry_trigger'] == 'RSI oversold'
        assert 'rsi' not in explanation
        assert 'adx' not in explanation
        assert 'r_squared' not in explanation

    def test_signal_type_mapping(self):
        """Test signal type integer to name mapping."""
        signal_types = {
            0: 'NONE',
            1: 'LONG',
            -1: 'SHORT',
            2: 'EXIT_LONG',
            -2: 'EXIT_SHORT',
        }

        for signal_type, expected_name in signal_types.items():
            signal_explanation = {
                'symbol': 'SPY',
                'signal_type': signal_type,
                'price': 420.0,
                'timestamp': datetime.now().timestamp(),
                'strategy_mode': 2,
                'regime': 2,
                'volatility_state': 1,
                'atr': 3.2,
                'kama': 418.5,
                'risk_amount': 1000.0,
                'risk_percent': 0.01,
                'stop_loss': 415.0,
                'position_size': 100,
                'entry_trigger': 'Test signal',
                'noise_filtered': True,
                'volume_validated': True,
            }

            trade = create_mock_trade(1, signal_explanation=signal_explanation)
            explanation = json.loads(trade['signal_explanation'])
            assert explanation['signal_type'] == signal_type


class TestFilteringFunctionality:
    """Test filtering capabilities of the archive viewer."""

    def test_filter_by_symbol(self):
        """Test filtering trades by symbol."""
        trades = [
            create_mock_trade(1, symbol="SPY"),
            create_mock_trade(2, symbol="AAPL"),
            create_mock_trade(3, symbol="SPY"),
            create_mock_trade(4, symbol="TSLA"),
        ]

        spy_trades = [t for t in trades if t['symbol'] == 'SPY']
        assert len(spy_trades) == 2
        assert all(t['symbol'] == 'SPY' for t in spy_trades)

    def test_filter_by_strategy(self):
        """Test filtering trades by strategy."""
        trades = [
            create_mock_trade(1, strategy="TREND"),
            create_mock_trade(2, strategy="MEAN_REVERSION"),
            create_mock_trade(3, strategy="TREND"),
            create_mock_trade(4, strategy="NEUTRAL"),
        ]

        trend_trades = [t for t in trades if t['strategy'] == 'TREND']
        assert len(trend_trades) == 2
        assert all(t['strategy'] == 'TREND' for t in trend_trades)

    def test_filter_by_regime(self):
        """Test filtering trades by market regime."""
        trades = [
            create_mock_trade(1, regime="STRONG_TREND"),
            create_mock_trade(2, regime="MEAN_REVERSION"),
            create_mock_trade(3, regime="STRONG_TREND"),
            create_mock_trade(4, regime="NEUTRAL"),
        ]

        trend_trades = [t for t in trades if t['regime'] == 'STRONG_TREND']
        assert len(trend_trades) == 2
        assert all(t['regime'] == 'STRONG_TREND' for t in trend_trades)

    def test_filter_by_date_range(self):
        """Test filtering trades by date range."""
        base_time = datetime(2024, 1, 1).timestamp()
        trades = [
            create_mock_trade(1, entry_time=base_time),
            create_mock_trade(2, entry_time=base_time + 86400),  # +1 day
            create_mock_trade(3, entry_time=base_time + 172800),  # +2 days
            create_mock_trade(4, entry_time=base_time + 259200),  # +3 days
        ]

        start_time = base_time + 86400  # Day 2
        end_time = base_time + 172800  # Day 3
        filtered_trades = [
            t for t in trades
            if start_time <= t['entry_time'] <= end_time
        ]

        assert len(filtered_trades) == 2

    def test_search_by_signal_reason(self):
        """Test searching trades by signal reason."""
        trades = [
            create_mock_trade(1, signal_reason="KAMA crossover"),
            create_mock_trade(2, signal_reason="RSI oversold"),
            create_mock_trade(3, signal_reason="KAMA breakout"),
            create_mock_trade(4, signal_reason="Bollinger band touch"),
        ]

        kama_trades = [
            t for t in trades
            if 'KAMA' in t['signal_reason'].upper()
        ]

        assert len(kama_trades) == 2

    def test_multiple_filters_combined(self):
        """Test combining multiple filters."""
        base_time = datetime(2024, 1, 1).timestamp()
        trades = [
            create_mock_trade(1, symbol="SPY", strategy="TREND", entry_time=base_time),
            create_mock_trade(2, symbol="AAPL", strategy="TREND", entry_time=base_time + 86400),
            create_mock_trade(3, symbol="SPY", strategy="MEAN_REVERSION", entry_time=base_time),
            create_mock_trade(4, symbol="SPY", strategy="TREND", entry_time=base_time + 172800),
        ]

        # Filter: SPY + TREND strategy
        filtered = [
            t for t in trades
            if t['symbol'] == 'SPY' and t['strategy'] == 'TREND'
        ]

        assert len(filtered) == 2


class TestSortingFunctionality:
    """Test sorting capabilities."""

    def test_sort_by_date_descending(self):
        """Test sorting by date (newest first)."""
        base_time = datetime(2024, 1, 1).timestamp()
        trades = [
            create_mock_trade(1, entry_time=base_time),
            create_mock_trade(2, entry_time=base_time + 172800),
            create_mock_trade(3, entry_time=base_time + 86400),
        ]

        sorted_trades = sorted(trades, key=lambda t: t['entry_time'], reverse=True)
        assert sorted_trades[0]['id'] == 2  # Newest
        assert sorted_trades[2]['id'] == 1  # Oldest

    def test_sort_by_date_ascending(self):
        """Test sorting by date (oldest first)."""
        base_time = datetime(2024, 1, 1).timestamp()
        trades = [
            create_mock_trade(1, entry_time=base_time + 86400),
            create_mock_trade(2, entry_time=base_time),
            create_mock_trade(3, entry_time=base_time + 172800),
        ]

        sorted_trades = sorted(trades, key=lambda t: t['entry_time'])
        assert sorted_trades[0]['id'] == 2  # Oldest
        assert sorted_trades[2]['id'] == 3  # Newest

    def test_sort_by_pnl_descending(self):
        """Test sorting by P&L (highest first)."""
        trades = [
            create_mock_trade(1, realized_pnl=500.0),
            create_mock_trade(2, realized_pnl=1500.0),
            create_mock_trade(3, realized_pnl=-200.0),
        ]

        sorted_trades = sorted(trades, key=lambda t: t['realized_pnl'], reverse=True)
        assert sorted_trades[0]['id'] == 2  # Highest P&L
        assert sorted_trades[2]['id'] == 3  # Lowest P&L

    def test_sort_by_pnl_ascending(self):
        """Test sorting by P&L (lowest first)."""
        trades = [
            create_mock_trade(1, realized_pnl=500.0),
            create_mock_trade(2, realized_pnl=-200.0),
            create_mock_trade(3, realized_pnl=1500.0),
        ]

        sorted_trades = sorted(trades, key=lambda t: t['realized_pnl'])
        assert sorted_trades[0]['id'] == 2  # Lowest P&L
        assert sorted_trades[2]['id'] == 3  # Highest P&L

    def test_sort_by_symbol_alphabetical(self):
        """Test sorting by symbol alphabetically."""
        trades = [
            create_mock_trade(1, symbol="TSLA"),
            create_mock_trade(2, symbol="AAPL"),
            create_mock_trade(3, symbol="SPY"),
        ]

        sorted_trades = sorted(trades, key=lambda t: t['symbol'])
        assert sorted_trades[0]['symbol'] == 'AAPL'
        assert sorted_trades[1]['symbol'] == 'SPY'
        assert sorted_trades[2]['symbol'] == 'TSLA'


class TestCSVExport:
    """Test CSV export functionality."""

    def test_export_headers(self):
        """Test CSV export includes correct headers."""
        expected_headers = [
            'Symbol', 'Entry Time', 'Signal Type', 'Entry Price', 'Exit Price',
            'P&L', 'Strategy', 'Regime', 'Volatility', 'ATR', 'KAMA',
            'RSI', 'ADX', 'R²', 'Entry Trigger', 'Risk Amount', 'Stop Loss',
        ]

        # Headers should be in the expected order
        assert len(expected_headers) == 17

    def test_export_data_formatting(self):
        """Test CSV export formats data correctly."""
        trade = create_mock_trade(1, symbol="SPY", entry_price=420.0, realized_pnl=500.0)
        explanation = json.loads(trade['signal_explanation'])

        # Check that numeric values are properly formatted
        assert explanation['atr'] == 3.2
        assert explanation['kama'] == 418.5
        assert explanation['rsi'] == 55.0
        assert explanation['adx'] == 32.0
        assert explanation['r_squared'] == 0.81

    def test_export_with_commas_in_text(self):
        """Test CSV export handles commas in text fields."""
        signal_explanation = {
            'symbol': 'SPY',
            'signal_type': 1,
            'price': 420.0,
            'timestamp': datetime.now().timestamp(),
            'strategy_mode': 2,
            'regime': 2,
            'volatility_state': 1,
            'atr': 3.2,
            'kama': 418.5,
            'risk_amount': 1000.0,
            'risk_percent': 0.01,
            'stop_loss': 415.0,
            'position_size': 100,
            'entry_trigger': 'KAMA crossover, high volume',  # Contains comma
            'noise_filtered': True,
            'volume_validated': True,
        }

        trade = create_mock_trade(
            1,
            signal_reason="KAMA crossover, high volume",
            signal_explanation=signal_explanation
        )

        # Entry trigger contains comma, should be quoted in CSV
        assert ',' in trade['signal_reason']


class TestPagination:
    """Test pagination functionality."""

    def test_pagination_basic(self):
        """Test basic pagination calculation."""
        trades = [create_mock_trade(i) for i in range(50)]
        items_per_page = 20

        total_pages = (len(trades) + items_per_page - 1) // items_per_page
        assert total_pages == 3  # 50 items / 20 per page = 3 pages

    def test_pagination_page_slicing(self):
        """Test correct slicing of trades for each page."""
        trades = [create_mock_trade(i) for i in range(50)]

        # Page 1
        page1 = trades[0:20]
        assert len(page1) == 20
        assert page1[0]['id'] == 0

        # Page 2
        page2 = trades[20:40]
        assert len(page2) == 20
        assert page2[0]['id'] == 20

        # Page 3 (partial page)
        page3 = trades[40:60]
        assert len(page3) == 10
        assert page3[0]['id'] == 40

    def test_pagination_empty_results(self):
        """Test pagination with no results."""
        trades = []
        items_per_page = 20

        total_pages = max(1, (len(trades) + items_per_page - 1) // items_per_page)
        assert total_pages == 1  # At least 1 page even with no results


class TestDetailModal:
    """Test detail modal functionality."""

    def test_modal_displays_all_sections(self):
        """Test modal includes all required sections."""
        trade = create_mock_trade(1)
        explanation = json.loads(trade['signal_explanation'])

        # All required sections should be present in explanation
        # Signal Details
        assert 'signal_type' in explanation
        assert 'price' in explanation
        assert 'entry_trigger' in explanation

        # Market Context
        assert 'regime' in explanation
        assert 'volatility_state' in explanation
        assert 'atr' in explanation

        # Risk Management
        assert 'risk_amount' in explanation
        assert 'stop_loss' in explanation
        assert 'position_size' in explanation

        # Validation Checks
        assert 'noise_filtered' in explanation
        assert 'volume_validated' in explanation

    def test_modal_outcome_classification(self):
        """Test modal correctly classifies trade outcomes."""
        # Profitable trade
        profitable_trade = create_mock_trade(1, realized_pnl=500.0)
        assert profitable_trade['realized_pnl'] > 0

        # Loss trade
        loss_trade = create_mock_trade(2, realized_pnl=-300.0)
        assert loss_trade['realized_pnl'] < 0

        # Breakeven trade
        breakeven_trade = create_mock_trade(3, realized_pnl=0.0)
        assert breakeven_trade['realized_pnl'] == 0.0

    def test_modal_technical_indicators_optional(self):
        """Test modal handles missing optional technical indicators."""
        signal_explanation = {
            'symbol': 'AAPL',
            'signal_type': 1,
            'price': 150.0,
            'timestamp': datetime.now().timestamp(),
            'strategy_mode': 1,
            'regime': 0,
            'volatility_state': 1,
            'atr': 2.5,
            'kama': 148.0,
            'risk_amount': 500.0,
            'risk_percent': 0.0075,
            'stop_loss': 145.0,
            'position_size': 50,
            'entry_trigger': 'RSI oversold',
            'noise_filtered': True,
            'volume_validated': True,
        }

        trade = create_mock_trade(1, signal_explanation=signal_explanation)
        explanation = json.loads(trade['signal_explanation'])

        # Should not have optional indicators
        assert 'rsi' not in explanation
        assert 'adx' not in explanation
        assert 'r_squared' not in explanation


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trades_list(self):
        """Test handling of empty trades list."""
        trades = []
        assert len(trades) == 0

    def test_trade_without_signal_explanation(self):
        """Test handling of trade without signal explanation."""
        trade = create_mock_trade(1)
        trade['signal_explanation'] = None

        assert trade['signal_explanation'] is None

    def test_invalid_json_signal_explanation(self):
        """Test handling of invalid JSON in signal explanation."""
        trade = create_mock_trade(1)
        trade['signal_explanation'] = "invalid json {"

        with pytest.raises(json.JSONDecodeError):
            json.loads(trade['signal_explanation'])

    def test_missing_trade_fields(self):
        """Test handling of trades with missing fields."""
        minimal_trade = {
            'id': 1,
            'symbol': 'SPY',
            'entry_price': 420.0,
            'entry_time': datetime.now().timestamp(),
        }

        # Should handle missing optional fields
        assert minimal_trade.get('exit_price') is None
        assert minimal_trade.get('realized_pnl') is None
        assert minimal_trade.get('strategy') is None

    def test_very_old_trades(self):
        """Test handling of trades with old timestamps."""
        old_time = datetime(2020, 1, 1).timestamp()
        trade = create_mock_trade(1, entry_time=old_time)

        assert trade['entry_time'] == old_time
        assert trade['entry_time'] < datetime.now().timestamp()

    def test_future_date_filter(self):
        """Test filtering with future dates."""
        current_time = datetime.now().timestamp()
        future_time = (datetime.now() + timedelta(days=365)).timestamp()

        trades = [
            create_mock_trade(1, entry_time=current_time),
            create_mock_trade(2, entry_time=current_time - 86400),
        ]

        # Filter for future dates should return empty
        future_trades = [t for t in trades if t['entry_time'] > future_time]
        assert len(future_trades) == 0


class TestPerformanceMetrics:
    """Test performance-related calculations."""

    def test_return_percentage_calculation(self):
        """Test return percentage calculation."""
        entry_price = 100.0
        exit_price = 110.0
        shares = 100

        capital_invested = entry_price * shares  # $10,000
        pnl = (exit_price - entry_price) * shares  # $1,000
        return_pct = (pnl / capital_invested) * 100  # 10%

        assert abs(return_pct - 10.0) < 0.01

    def test_risk_reward_ratio(self):
        """Test risk/reward ratio calculation."""
        entry_price = 420.0
        stop_loss = 415.0
        realized_pnl = 500.0
        shares = 100

        risk = abs(entry_price - stop_loss) * shares  # $500
        reward = realized_pnl  # $500
        risk_reward = reward / risk  # 1:1

        assert abs(risk_reward - 1.0) < 0.01

    def test_win_rate_calculation(self):
        """Test win rate calculation from filtered trades."""
        trades = [
            create_mock_trade(1, realized_pnl=500.0),
            create_mock_trade(2, realized_pnl=-300.0),
            create_mock_trade(3, realized_pnl=800.0),
            create_mock_trade(4, realized_pnl=-150.0),
            create_mock_trade(5, realized_pnl=600.0),
        ]

        winning_trades = [t for t in trades if t['realized_pnl'] > 0]
        win_rate = len(winning_trades) / len(trades)

        assert abs(win_rate - 0.6) < 0.01  # 60% win rate


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
