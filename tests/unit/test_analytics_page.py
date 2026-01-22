"""
Unit tests for the Analytics page implementation.

This module validates:
- Analytics page file exists
- Required dependencies are installed (lightweight-charts)
- Page structure and components
- All required features implemented
"""

import json
from pathlib import Path


def test_analytics_page_exists():
    """Verify analytics page file exists."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    assert analytics_page.exists(), "Analytics page should exist at app/analytics/page.tsx"


def test_analytics_page_has_required_imports():
    """Verify analytics page imports required dependencies."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for React imports
    assert "import React" in content, "Should import React"
    assert "useEffect" in content, "Should use useEffect hook"
    assert "useRef" in content, "Should use useRef hook"
    assert "useState" in content, "Should use useState hook"

    # Check for lightweight-charts imports
    assert "from 'lightweight-charts'" in content, "Should import from lightweight-charts"
    assert "createChart" in content, "Should import createChart"
    assert "IChartApi" in content, "Should import IChartApi type"


def test_analytics_page_has_chart_initialization():
    """Verify chart initialization logic is present."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for chart creation
    assert "createChart" in content, "Should create chart instance"
    assert "chartContainerRef" in content, "Should have chart container ref"
    assert "chartRef" in content, "Should have chart ref"

    # Check for chart configuration
    assert "layout" in content, "Should configure layout"
    assert "background" in content, "Should configure background"
    assert "grid" in content, "Should configure grid"


def test_analytics_page_has_candlestick_series():
    """Verify candlestick series is configured."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for candlestick series
    assert "addCandlestickSeries" in content, "Should add candlestick series"
    assert "upColor" in content, "Should configure upColor"
    assert "downColor" in content, "Should configure downColor"
    assert "wickUpColor" in content, "Should configure wickUpColor"
    assert "wickDownColor" in content, "Should configure wickDownColor"


def test_analytics_page_has_kama_overlay():
    """Verify KAMA line overlay is implemented."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for KAMA series
    assert "kamaSeries" in content or "KAMA" in content, "Should have KAMA series"
    assert "addLineSeries" in content, "Should add line series for KAMA"

    # Check for KAMA data
    assert "kama" in content.lower(), "Should reference KAMA data"


def test_analytics_page_has_atr_bands():
    """Verify ATR bands overlay is implemented."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for ATR band series
    assert "atr" in content.lower(), "Should reference ATR"
    assert (
        "atrUpper" in content or "atr_upper" in content.lower() or "ATR Upper" in content
    ), "Should have ATR upper band"
    assert (
        "atrLower" in content or "atr_lower" in content.lower() or "ATR Lower" in content
    ), "Should have ATR lower band"


def test_analytics_page_has_signal_markers():
    """Verify buy/sell signal annotations are implemented."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for signal markers
    assert "signal" in content.lower(), "Should reference signals"
    assert "marker" in content.lower(), "Should use markers"

    # Check for marker types
    assert (
        "arrowUp" in content or "arrow_up" in content.lower() or "Buy" in content
    ), "Should have buy signals"
    assert (
        "arrowDown" in content or "arrow_down" in content.lower() or "Sell" in content
    ), "Should have sell signals"


def test_analytics_page_has_indicator_panel():
    """Verify indicator panel showing ATR, RSI, ADX, regime state."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for all required indicators
    assert "ATR" in content, "Should display ATR indicator"
    assert "RSI" in content, "Should display RSI indicator"
    assert "ADX" in content, "Should display ADX indicator"
    assert "regime" in content.lower() or "Regime" in content, "Should display regime state"


def test_analytics_page_has_performance_metrics():
    """Verify performance metrics display (total return %, Sharpe, win rate, max drawdown)."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for all required metrics
    assert (
        "totalReturn" in content or "total_return" in content.lower() or "Total Return" in content
    ), "Should display total return"
    assert (
        "sharpe" in content.lower() or "Sharpe" in content
    ), "Should display Sharpe ratio"
    assert (
        "winRate" in content or "win_rate" in content.lower() or "Win Rate" in content
    ), "Should display win rate"
    assert (
        "maxDrawdown" in content
        or "max_drawdown" in content.lower()
        or "Max Drawdown" in content
        or "drawdown" in content.lower()
    ), "Should display max drawdown"


def test_analytics_page_has_symbol_selector():
    """Verify symbol selection controls."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for symbol selector
    assert "symbol" in content.lower(), "Should have symbol selection"
    assert "select" in content.lower() or "option" in content.lower(), "Should have select control"


def test_analytics_page_has_timeframe_selector():
    """Verify timeframe selection controls."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for timeframe selector
    assert "timeframe" in content.lower() or "interval" in content.lower(), "Should have timeframe selection"


def test_analytics_page_has_responsive_design():
    """Verify responsive design classes are used."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for responsive design patterns
    assert (
        "md:" in content or "lg:" in content or "responsive" in content.lower()
    ), "Should have responsive design classes"


def test_analytics_page_has_loading_state():
    """Verify loading state is implemented."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for loading state
    assert "loading" in content.lower(), "Should have loading state"
    assert "Loading" in content or "loading" in content, "Should display loading indicator"


def test_analytics_page_has_color_coding():
    """Verify color coding for profit/loss and indicators."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for color coding
    assert (
        "green" in content.lower() or "#26a69a" in content.lower() or "text-green" in content
    ), "Should use green for positive values"
    assert (
        "red" in content.lower() or "#ef5350" in content.lower() or "text-red" in content
    ), "Should use red for negative values"


def test_analytics_page_has_chart_resize_handler():
    """Verify chart resize handling is implemented."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for resize handling
    assert "resize" in content.lower(), "Should handle window resize"
    assert "addEventListener" in content or "resize" in content, "Should add resize listener"


def test_analytics_page_has_cleanup():
    """Verify proper cleanup in useEffect."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for cleanup
    assert "return () =>" in content or "return function" in content, "Should have cleanup function"
    assert "removeEventListener" in content or "remove()" in content, "Should clean up resources"


def test_lightweight_charts_dependency():
    """Verify lightweight-charts is in package.json dependencies."""
    package_json = Path("frontend/package.json")
    assert package_json.exists(), "package.json should exist"

    content = json.loads(package_json.read_text())
    dependencies = content.get("dependencies", {})

    assert (
        "lightweight-charts" in dependencies
    ), "lightweight-charts should be in dependencies"


def test_analytics_page_typescript_types():
    """Verify TypeScript type definitions are used."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for TypeScript types
    assert "interface" in content or "type " in content, "Should define TypeScript types"
    assert ": " in content, "Should use type annotations"


def test_analytics_page_has_proper_structure():
    """Verify analytics page has proper component structure."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for main component export
    assert "export default" in content, "Should export default component"
    assert "function" in content or "const" in content, "Should define component function"

    # Check for return statement with JSX
    assert "return" in content, "Should have return statement"
    assert "<div" in content, "Should render JSX"


def test_analytics_page_has_dark_theme():
    """Verify dark theme styling is implemented."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for dark theme colors
    assert (
        "bg-gray" in content
        or "background: { color: '#" in content
        or "dark" in content.lower()
    ), "Should have dark theme styling"


def test_analytics_page_has_grid_layout():
    """Verify grid layout is used for metrics display."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for grid layout
    assert (
        "grid" in content.lower() or "flex" in content.lower()
    ), "Should use grid or flex layout"


def test_analytics_page_client_component():
    """Verify page is marked as client component for Next.js."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for 'use client' directive
    assert "'use client'" in content or '"use client"' in content, "Should be marked as client component"


def test_analytics_page_auto_refresh():
    """Verify auto-refresh functionality is implemented."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for auto-refresh with interval
    assert "setInterval" in content or "interval" in content.lower(), "Should have auto-refresh"
    assert "clearInterval" in content or "clear" in content.lower(), "Should clear interval on cleanup"


def test_analytics_page_has_header():
    """Verify page has proper header with title and description."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for header elements
    assert "Analytics" in content or "analytics" in content.lower(), "Should have analytics title"
    assert "h1" in content.lower() or "heading" in content.lower(), "Should have heading element"


def test_analytics_page_conditional_rendering():
    """Verify conditional rendering based on state."""
    analytics_page = Path("frontend/app/analytics/page.tsx")
    content = analytics_page.read_text()

    # Check for conditional rendering
    assert "{" in content and "?" in content and ":" in content, "Should use ternary operators"
    assert "&&" in content or "||" in content, "Should use logical operators"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
