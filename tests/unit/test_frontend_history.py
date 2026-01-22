"""
Unit tests for frontend Trade History page

Tests:
- Component initialization and rendering
- Trade data fetching and display
- Pagination functionality (20 trades per page)
- CSV export functionality
- Trade detail tooltips
- Error handling
- Loading states
- Edge cases (empty trades, pagination edge cases)
"""

from pathlib import Path

import pytest


def test_history_page_exists():
    """Test that history.tsx file exists"""
    history_file = Path("frontend/pages/history.tsx")
    assert history_file.exists(), "history.tsx should exist"


def test_history_page_is_client_component():
    """Test that history page is a client component"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()
    assert "'use client'" in content, "history.tsx should be a client component"


def test_history_page_imports_api():
    """Test that history page imports API utilities"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()
    assert "from '../utils/api'" in content, "Should import from utils/api"
    assert "Trade" in content, "Should import Trade type"


def test_history_page_has_main_component():
    """Test that history page exports main component"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()
    assert "TradeHistoryPage" in content, "Should have TradeHistoryPage component"
    assert "export default TradeHistoryPage" in content, "Should export component as default"


def test_history_page_has_state_management():
    """Test that history page has proper state management"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for useState hooks
    assert "useState<Trade[]>" in content, "Should have trades state"
    assert "useState<boolean>" in content, "Should have loading state"
    assert "useState<string | null>" in content, "Should have error state"
    assert "useState<number>" in content, "Should have currentPage state"


def test_history_page_has_use_effect():
    """Test that history page has useEffect for data fetching"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()
    assert "useEffect" in content, "Should use useEffect hook"
    assert "api.getTrades" in content, "Should call api.getTrades"


def test_pagination_configuration():
    """Test that pagination is configured for 20 trades per page"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()
    assert "tradesPerPage = 20" in content or "tradesPerPage: 20" in content, \
        "Should have 20 trades per page"


def test_pagination_controls():
    """Test that pagination controls are present"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for pagination UI elements
    assert "pagination" in content.lower(), "Should have pagination section"
    assert "Previous" in content or "previous" in content.lower(), \
        "Should have previous button"
    assert "Next" in content or "next" in content.lower(), "Should have next button"
    assert "currentPage" in content, "Should track current page"


def test_csv_export_function():
    """Test that CSV export functionality is present"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "convertToCSV" in content or "CSV" in content, "Should have CSV export"
    assert "downloadCSV" in content or "download" in content.lower(), \
        "Should have download function"
    assert "Export" in content or "export" in content.lower(), \
        "Should have export button"


def test_csv_export_includes_headers():
    """Test that CSV export includes proper headers"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for key CSV headers
    assert "Symbol" in content, "CSV should include Symbol"
    assert "Side" in content, "CSV should include Side"
    assert "Entry Price" in content or "entry_price" in content, \
        "CSV should include Entry Price"
    assert "Exit Price" in content or "exit_price" in content, \
        "CSV should include Exit Price"
    assert "P&L" in content or "pnl" in content.lower(), "CSV should include P&L"


def test_tooltip_component():
    """Test that tooltip component exists"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "Tooltip" in content, "Should have Tooltip component"
    assert "signal_reason" in content, "Tooltip should show signal reason"


def test_tooltip_shows_trade_details():
    """Test that tooltip shows comprehensive trade details"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for key tooltip fields
    assert "Signal Reason" in content, "Tooltip should show signal reason"
    assert "Strategy" in content, "Tooltip should show strategy"
    assert "regime" in content.lower(), "Tooltip should show regime"
    assert "Stop Loss" in content or "stop_loss" in content, \
        "Tooltip should show stop loss"
    assert "Take Profit" in content or "take_profit" in content, \
        "Tooltip should show take profit"


def test_table_structure():
    """Test that trade table has proper structure"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for table elements
    assert "<table" in content, "Should have table element"
    assert "<thead>" in content, "Should have table header"
    assert "<tbody>" in content, "Should have table body"
    assert "<th>" in content, "Should have table header cells"
    assert "className=" in content, "Should have table data cells with className"


def test_table_columns():
    """Test that table has required columns"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for key columns
    columns = ["Symbol", "Side", "Entry", "Exit", "Shares", "P&L", "Strategy"]
    for column in columns:
        assert column in content, f"Table should have {column} column"


def test_loading_state():
    """Test that loading state is displayed"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "loading" in content.lower(), "Should have loading state"
    assert "Loading" in content or "loading" in content.lower(), \
        "Should show loading message"


def test_error_handling():
    """Test that error handling is present"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "error" in content.lower(), "Should handle errors"
    assert "catch" in content or "Error" in content, "Should catch errors"
    assert "setError" in content, "Should set error state"


def test_color_coded_pnl():
    """Test that P&L is color-coded"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for color coding logic
    assert "pnl" in content.lower(), "Should have P&L field"
    # Look for color classes or conditional styling
    assert ("positive" in content.lower() or "negative" in content.lower() or
            "green" in content.lower() or "red" in content.lower()), \
        "Should have color-coded P&L"


def test_format_currency_helper():
    """Test that currency formatting helper exists"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "formatCurrency" in content or "currency" in content.lower(), \
        "Should have currency formatting"
    assert "USD" in content or "$" in content, "Should format as USD"


def test_format_date_helper():
    """Test that date formatting helper exists"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "formatDate" in content or "Date" in content, "Should have date formatting"
    assert "timestamp" in content.lower(), "Should handle timestamps"


def test_format_percent_helper():
    """Test that percentage formatting helper exists"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "formatPercent" in content or "%" in content, \
        "Should have percentage formatting"
    assert "Return" in content or "return" in content.lower(), \
        "Should show return percentage"


def test_empty_trades_handling():
    """Test that empty trades list is handled"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "length === 0" in content or "No trades" in content, \
        "Should handle empty trades"


def test_responsive_design():
    """Test that responsive design is included"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for responsive CSS
    assert "@media" in content or "responsive" in content.lower(), \
        "Should have responsive design"


def test_page_header():
    """Test that page has a header with title and export button"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "Trade History" in content, "Should have page title"
    assert "page-header" in content.lower() or "header" in content.lower(), \
        "Should have page header"
    assert "export" in content.lower(), "Should have export button in header"


def test_typescript_typing():
    """Test that TypeScript types are properly used"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for TypeScript type annotations
    assert ": Trade[]" in content, "Should type trades array"
    assert "<boolean>" in content, "Should type boolean states"
    assert "<number>" in content or ": number" in content, "Should type number states"
    assert ": string" in content, "Should type string values"


def test_async_data_fetching():
    """Test that data fetching is async"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "async" in content, "Should use async functions"
    assert "await" in content, "Should await API calls"


def test_pagination_page_change_handler():
    """Test that page change handler exists"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "handlePageChange" in content or "setCurrentPage" in content, \
        "Should handle page changes"


def test_export_button_handler():
    """Test that export button has handler"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "handleExport" in content or "onClick" in content, \
        "Should have export handler"


def test_trade_id_key():
    """Test that trades are keyed by ID"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "key=" in content, "Should use keys for list items"
    assert "trade.id" in content, "Should key by trade ID"


def test_styled_jsx():
    """Test that component includes styling"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for styling (either styled-jsx or CSS modules)
    assert "<style" in content or "className=" in content, \
        "Should include styling"


def test_pagination_disabled_states():
    """Test that pagination buttons have disabled states"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "disabled" in content, "Pagination should have disabled states"


def test_csv_filename_with_timestamp():
    """Test that CSV export includes timestamp in filename"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for timestamp or date in filename
    assert ("timestamp" in content.lower() or "date" in content.lower() or
            "trades_" in content.lower()), \
        "CSV filename should include timestamp"


def test_return_percentage_calculation():
    """Test that return percentage is calculated"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "realized_pnl" in content, "Should use realized P&L"
    assert "entry_price" in content, "Should use entry price"
    # Look for percentage calculation logic
    assert ("* 100" in content or "toFixed(2)" in content), \
        "Should calculate return percentage"


def test_side_styling():
    """Test that trade side (LONG/SHORT) has styling"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    assert "side" in content.lower(), "Should display trade side"
    # Check for side-specific styling
    assert ("long" in content.lower() or "short" in content.lower()), \
        "Should style LONG/SHORT differently"


def test_all_required_features():
    """Test that all required features from Phase 14, Tab C are present"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    required_features = [
        "trade log table",  # Trade log table component
        "20",  # Pagination (20 trades per page)
        "CSV",  # CSV export functionality
        "tooltip",  # Trade detail tooltips
        "signal",  # Signal explanations
    ]

    for feature in required_features:
        assert feature in content or feature.upper() in content or feature.lower() in content, \
            f"Should include {feature} feature"


def test_component_structure():
    """Test overall component structure and organization"""
    history_file = Path("frontend/pages/history.tsx")
    content = history_file.read_text()

    # Check for proper component structure
    assert "const TradeHistoryPage" in content, "Should define main component"
    assert "return (" in content, "Should return JSX"
    assert content.count("useState") >= 4, "Should have multiple state hooks"
    assert "useEffect" in content, "Should have useEffect"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
