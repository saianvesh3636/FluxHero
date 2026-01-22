"""
Unit tests for frontend backtest page functionality.

This test file validates the backtest page implementation including:
- Configuration form with all required parameters
- Date pickers for start/end dates
- Symbol selector with multiple options
- Parameter sliders for strategy and risk settings
- Run backtest button with loading state
- Results modal with performance metrics
- CSV export functionality
- Success criteria validation display
"""

import os
import unittest


class TestBacktestPage(unittest.TestCase):
    """Tests for the backtest page implementation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.backtest_page_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            '',
            'frontend',
            'app',
            'backtest',
            'page.tsx'
        )

    def test_backtest_page_exists(self):
        """Test that the backtest page file exists."""
        self.assertTrue(
            os.path.exists(self.backtest_page_path),
            "Backtest page should exist at app/backtest/page.tsx"
        )

    def test_backtest_page_is_client_component(self):
        """Test that the page uses 'use client' directive."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        self.assertIn(
            "'use client'",
            content,
            "Backtest page should be a client component"
        )

    def test_backtest_config_interface_defined(self):
        """Test that BacktestConfig interface is properly defined."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for BacktestConfig interface
        self.assertIn('interface BacktestConfig', content)

        # Check for required fields
        required_fields = [
            'symbol', 'startDate', 'endDate', 'initialCapital',
            'commission', 'slippage', 'emaPeriod', 'rsiPeriod',
            'atrPeriod', 'kamaPeriod', 'maxPositionSize', 'stopLossPct'
        ]

        for field in required_fields:
            self.assertIn(
                f'{field}:',
                content,
                f"BacktestConfig should include {field} field"
            )

    def test_backtest_result_interface_defined(self):
        """Test that BacktestResult interface is properly defined."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for BacktestResult interface
        self.assertIn('interface BacktestResult', content)

        # Check for required metrics
        required_metrics = [
            'totalReturn', 'totalReturnPct', 'sharpeRatio',
            'maxDrawdown', 'winRate', 'totalTrades',
            'avgWin', 'avgLoss', 'profitFactor'
        ]

        for metric in required_metrics:
            self.assertIn(
                f'{metric}:',
                content,
                f"BacktestResult should include {metric} field"
            )

    def test_symbol_selector_present(self):
        """Test that symbol selector dropdown exists."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for symbol selector
        self.assertIn('Symbol', content)
        self.assertIn('value={config.symbol}', content)

        # Check for multiple symbol options
        symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'MSFT']
        for symbol in symbols:
            self.assertIn(
                symbol,
                content,
                f"Symbol selector should include {symbol} option"
            )

    def test_date_pickers_present(self):
        """Test that start and end date pickers exist."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for date inputs
        self.assertIn('Start Date', content)
        self.assertIn('End Date', content)
        self.assertIn('type="date"', content)
        self.assertIn('value={config.startDate}', content)
        self.assertIn('value={config.endDate}', content)

    def test_initial_capital_input_present(self):
        """Test that initial capital input exists."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        self.assertIn('Initial Capital', content)
        self.assertIn('value={config.initialCapital}', content)
        self.assertIn('type="number"', content)

    def test_commission_slippage_inputs_present(self):
        """Test that commission and slippage inputs exist."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        self.assertIn('Commission', content)
        self.assertIn('Slippage', content)
        self.assertIn('value={config.commission}', content)
        self.assertIn('value={config.slippage}', content)

    def test_strategy_parameter_sliders_present(self):
        """Test that strategy parameter sliders exist."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for parameter labels
        parameters = ['EMA Period', 'RSI Period', 'ATR Period', 'KAMA Period']
        for param in parameters:
            self.assertIn(
                param,
                content,
                f"Strategy parameters should include {param}"
            )

        # Check for slider inputs
        slider_configs = [
            'value={config.emaPeriod}',
            'value={config.rsiPeriod}',
            'value={config.atrPeriod}',
            'value={config.kamaPeriod}'
        ]
        for config in slider_configs:
            self.assertIn(
                config,
                content,
                f"Strategy slider {config} should be present"
            )

        # Check for range type
        self.assertIn('type="range"', content)

    def test_risk_parameter_sliders_present(self):
        """Test that risk parameter sliders exist."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for risk parameters
        self.assertIn('Max Position Size', content)
        self.assertIn('Stop Loss', content)
        self.assertIn('value={config.maxPositionSize}', content)
        self.assertIn('value={config.stopLossPct}', content)

    def test_run_backtest_button_present(self):
        """Test that run backtest button exists."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        self.assertIn('Run Backtest', content)
        self.assertIn('onClick={runBacktest}', content)
        # LoadingButton component handles disabled state internally via isLoading prop
        self.assertIn('isLoading={isRunning}', content)

    def test_loading_spinner_present(self):
        """Test that loading spinner is implemented."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for loading state
        self.assertIn('isRunning', content)
        self.assertIn('Running Backtest', content)
        # LoadingButton component contains animate-spin
        self.assertIn('LoadingButton', content)
        self.assertIn('isLoading={isRunning}', content)

    def test_api_call_implementation(self):
        """Test that API call to backend is implemented."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for API call
        self.assertIn("fetch('/api/backtest'", content)
        self.assertIn("method: 'POST'", content)
        self.assertIn("'Content-Type': 'application/json'", content)
        self.assertIn('body: JSON.stringify(config)', content)

    def test_error_handling_present(self):
        """Test that error handling is implemented."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        self.assertIn('setError', content)
        self.assertIn('try', content)
        self.assertIn('catch', content)
        self.assertIn('finally', content)

    def test_results_modal_present(self):
        """Test that results modal is implemented."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for modal structure
        self.assertIn('showResults', content)
        self.assertIn('results', content)
        self.assertIn('Backtest Results', content)

        # Check for modal overlay
        self.assertIn('fixed inset-0', content)
        self.assertIn('bg-opacity-75', content)

    def test_performance_metrics_display(self):
        """Test that performance metrics are displayed in results."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for metric displays
        metrics = [
            'totalReturn', 'totalReturnPct', 'sharpeRatio',
            'maxDrawdown', 'winRate', 'profitFactor'
        ]

        for metric in metrics:
            self.assertIn(
                f'results.{metric}',
                content,
                f"Results should display {metric}"
            )

    def test_success_criteria_validation_display(self):
        """Test that success criteria validation is displayed."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for success criteria section
        self.assertIn('Success Criteria', content)

        # Check for specific criteria (may be HTML encoded)
        self.assertTrue(
            'Sharpe' in content and '0.8' in content,
            "Should display Sharpe ratio criteria"
        )
        self.assertTrue(
            'DD' in content or 'Drawdown' in content,
            "Should display drawdown criteria"
        )
        self.assertTrue(
            'Win Rate' in content and '45' in content,
            "Should display win rate criteria"
        )

        # Check for PASS/FAIL logic
        self.assertIn('PASS', content)
        self.assertIn('FAIL', content)

    def test_trade_log_table_present(self):
        """Test that trade log table is displayed."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for trade log section
        self.assertIn('Trade Log', content)

        # Check for table structure
        self.assertIn('<table', content)
        self.assertIn('<thead', content)
        self.assertIn('<tbody', content)

        # Check for table columns
        columns = ['Entry Date', 'Exit Date', 'Side', 'Entry', 'Exit', 'Shares', 'P&L']
        for column in columns:
            self.assertIn(
                column,
                content,
                f"Trade log should include {column} column"
            )

    def test_csv_export_functionality(self):
        """Test that CSV export functionality is implemented."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for export function
        self.assertIn('exportToCSV', content)
        self.assertIn('Export Trade Log', content)

        # Check for CSV generation logic
        self.assertIn('text/csv', content)
        self.assertIn('download', content)
        self.assertIn('.csv', content)

    def test_color_coding_for_metrics(self):
        """Test that metrics have color coding (green/red for profit/loss)."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for color coding logic
        self.assertIn('text-green-400', content)
        self.assertIn('text-red-400', content)
        self.assertIn('text-yellow-400', content)

    def test_close_results_button_present(self):
        """Test that close button for results modal exists."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        self.assertIn('closeResults', content)
        self.assertIn('onClick={closeResults}', content)
        self.assertIn('Close', content)

    def test_default_config_values_set(self):
        """Test that default configuration values are set."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for useState with config
        self.assertIn('useState<BacktestConfig>', content)

        # Check for some default values
        defaults = ['SPY', '100000', '0.005', '0.01', '20', '14', '10', '3.0']
        for default in defaults:
            self.assertIn(
                default,
                content,
                f"Default value {default} should be set"
            )

    def test_responsive_design_classes(self):
        """Test that responsive design classes are used."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for responsive grid classes
        responsive_classes = [
            'md:grid-cols-2', 'lg:grid-cols-3', 'lg:grid-cols-4', 'lg:grid-cols-5'
        ]
        for cls in responsive_classes:
            self.assertIn(
                cls,
                content,
                f"Responsive class {cls} should be present"
            )

    def test_accessibility_labels(self):
        """Test that form inputs have proper labels."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for label tags
        label_count = content.count('<label')
        self.assertGreater(
            label_count,
            10,
            "Form should have multiple labels for accessibility"
        )

        # Check for className on labels
        self.assertIn('text-sm font-medium', content)

    def test_focus_styles_present(self):
        """Test that focus styles are implemented for accessibility."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for focus styles
        self.assertIn('focus:outline-none', content)
        self.assertIn('focus:ring-2', content)
        self.assertIn('focus:ring-blue-500', content)

    def test_modal_z_index_high(self):
        """Test that modal has high z-index for proper layering."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for z-index on modal
        self.assertIn('z-50', content)

    def test_modal_overflow_handling(self):
        """Test that modal has overflow handling for long results."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for overflow handling
        self.assertIn('overflow-y-auto', content)
        self.assertIn('max-h-', content)

    def test_file_structure_valid_typescript(self):
        """Test that the file has valid TypeScript structure."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for TypeScript features
        self.assertIn('interface', content)
        self.assertIn(': string', content)
        self.assertIn(': number', content)
        self.assertTrue(
            'boolean' in content,
            "File should use boolean type"
        )

        # Check for React hooks
        self.assertIn('useState', content)

        # Check for export default
        self.assertIn('export default function BacktestPage', content)

    def test_all_sections_present(self):
        """Test that all major sections are present."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        sections = [
            'Backtesting Module',
            'Backtest Configuration',
            'Strategy Parameters',
            'Risk Parameters',
            'Run Backtest',
            'Backtest Results'
        ]

        for section in sections:
            self.assertIn(
                section,
                content,
                f"Page should include '{section}' section"
            )


class TestBacktestPageIntegration(unittest.TestCase):
    """Integration tests for backtest page functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.backtest_page_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            '',
            'frontend',
            'app',
            'backtest',
            'page.tsx'
        )

    def test_config_change_handler_present(self):
        """Test that configuration change handler is implemented."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        self.assertIn('handleConfigChange', content)
        self.assertIn('setConfig', content)

    def test_run_backtest_function_present(self):
        """Test that runBacktest function is implemented."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for runBacktest function
        self.assertIn('const runBacktest', content)
        self.assertIn('setIsRunning(true)', content)
        self.assertIn('setIsRunning(false)', content)

    def test_state_management_complete(self):
        """Test that all necessary state variables are managed."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        states = ['config', 'isRunning', 'results', 'showResults', 'error']

        for state in states:
            self.assertIn(
                f'{state},',
                content,
                f"State variable '{state}' should be managed"
            )

    def test_min_max_values_set_on_sliders(self):
        """Test that sliders have min and max values."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for min/max attributes
        self.assertIn('min="', content)
        self.assertIn('max="', content)

    def test_step_values_set_where_needed(self):
        """Test that step values are set for decimal inputs."""
        with open(self.backtest_page_path) as f:
            content = f.read()

        # Check for step attributes
        self.assertIn('step="', content)


if __name__ == '__main__':
    unittest.main()
