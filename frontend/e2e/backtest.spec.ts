import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/api-mocks';

test.describe('Backtest Page', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page);
    await page.goto('/backtest');
  });

  test('should load backtest configuration form', async ({ page }) => {
    // Check page title
    await expect(page.locator('h1:has-text("Backtesting Module")')).toBeVisible();

    // Check form elements are present
    await expect(page.locator('select').first()).toBeVisible();
    await expect(page.locator('input[type="date"]').first()).toBeVisible();
    await expect(page.locator('button:has-text("Run Backtest")')).toBeVisible();
  });

  test('should allow changing configuration values', async ({ page }) => {
    // Change symbol
    await page.selectOption('select', 'AAPL');
    const selectedSymbol = await page.locator('select').inputValue();
    expect(selectedSymbol).toBe('AAPL');

    // Change initial capital
    const capitalInput = page.locator('input[type="number"]').first();
    await capitalInput.fill('50000');
    const capitalValue = await capitalInput.inputValue();
    expect(capitalValue).toBe('50000');

    // Adjust a slider (EMA Period)
    const emaSlider = page.locator('input[type="range"]').first();
    await emaSlider.fill('30');
    const emaValue = await emaSlider.inputValue();
    expect(emaValue).toBe('30');
  });

  test('should submit backtest and show results on success', async ({ page }) => {
    // Mock the API response
    await page.route('/api/backtest', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          totalReturn: 15000,
          totalReturnPct: 15.0,
          sharpeRatio: 1.2,
          maxDrawdown: 18.5,
          winRate: 52.3,
          totalTrades: 42,
          avgWin: 450.25,
          avgLoss: -280.50,
          profitFactor: 2.1,
          equity_curve: [100000, 102000, 103500, 105000],
          trade_log: [
            {
              entry_date: '2023-01-15',
              exit_date: '2023-01-20',
              symbol: 'SPY',
              side: 'LONG',
              entry_price: 400.50,
              exit_price: 405.25,
              shares: 100,
              pnl: 475.00,
            },
          ],
        }),
      });
    });

    // Click run backtest button
    const runButton = page.locator('button:has-text("Run Backtest")');
    await runButton.click();

    // Small delay for loading state to show
    await page.waitForTimeout(100);

    // Wait for results modal to appear
    await expect(page.locator('h2:has-text("Backtest Results")')).toBeVisible({ timeout: 10000 });

    // Verify results are displayed (use .first() to avoid strict mode violation)
    await expect(page.locator('text=/Total Return/i').first()).toBeVisible();
    await expect(page.locator('text=/Sharpe Ratio/i').first()).toBeVisible();
    await expect(page.locator('text=/Max Drawdown/i').first()).toBeVisible();
    await expect(page.locator('text=/Win Rate/i').first()).toBeVisible();

    // Verify trade log table is shown
    await expect(page.locator('table')).toBeVisible();

    // Verify at least one row in the table (trade log data)
    const tableRows = page.locator('table tbody tr');
    const rowCount = await tableRows.count();
    expect(rowCount).toBeGreaterThan(0);

    // Close modal
    const closeButton = page.locator('button:has-text("Close")').last();
    await closeButton.click();

    // Modal should be hidden
    await expect(page.locator('h2:has-text("Backtest Results")')).not.toBeVisible();
  });

  test('should show error state when backtest fails', async ({ page }) => {
    // Mock the API to return an error
    await page.route('/api/backtest', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    // Click run backtest button
    await page.locator('button:has-text("Run Backtest")').click();

    // Wait for error message to appear
    await expect(page.locator('text=/Backtest failed/i')).toBeVisible({ timeout: 10000 });

    // Verify retry button is available
    const retryButton = page.locator('button:has-text("Retry")');
    await expect(retryButton).toBeVisible();
  });

  test('should show error when backend is offline', async ({ page }) => {
    // Mock network failure
    await page.route('/api/backtest', async (route) => {
      await route.abort('failed');
    });

    // Click run backtest button
    await page.locator('button:has-text("Run Backtest")').click();

    // Wait for error to appear
    await expect(page.locator('text=/Error/i')).toBeVisible({ timeout: 10000 });

    // Verify retry button exists
    await expect(page.locator('button:has-text("Retry")').first()).toBeVisible();
  });

  test('should display all configuration sections', async ({ page }) => {
    // Check main sections - new design uses h3 for section headers
    await expect(page.locator('h3:has-text("Strategy Parameters")')).toBeVisible();
    await expect(page.locator('h3:has-text("Risk Parameters")')).toBeVisible();

    // Verify key input fields exist
    await expect(page.locator('label:has-text("Symbol")')).toBeVisible();
    await expect(page.locator('label:has-text("Start Date")')).toBeVisible();
    await expect(page.locator('label:has-text("End Date")')).toBeVisible();
    await expect(page.locator('label:has-text("Initial Capital")')).toBeVisible();
    await expect(page.locator('label:has-text("EMA Period")')).toBeVisible();
    await expect(page.locator('label:has-text("Max Position Size")')).toBeVisible();
  });

  test('should be responsive', async ({ page }) => {
    // Mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('h1:has-text("Backtesting Module")')).toBeVisible();

    // Tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('h1:has-text("Backtesting Module")')).toBeVisible();

    // Desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(page.locator('h1:has-text("Backtesting Module")')).toBeVisible();
  });

  test('should allow exporting trade log when results are shown', async ({ page }) => {
    // Mock successful backtest
    await page.route('/api/backtest', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          totalReturn: 15000,
          totalReturnPct: 15.0,
          sharpeRatio: 1.2,
          maxDrawdown: 18.5,
          winRate: 52.3,
          totalTrades: 42,
          avgWin: 450.25,
          avgLoss: -280.50,
          profitFactor: 2.1,
          equity_curve: [100000, 102000],
          trade_log: [
            {
              entry_date: '2023-01-15',
              exit_date: '2023-01-20',
              symbol: 'SPY',
              side: 'LONG',
              entry_price: 400.50,
              exit_price: 405.25,
              shares: 100,
              pnl: 475.00,
            },
          ],
        }),
      });
    });

    // Run backtest
    await page.locator('button:has-text("Run Backtest")').click();
    await expect(page.locator('h2:has-text("Backtest Results")')).toBeVisible({ timeout: 10000 });

    // Check export button exists
    const exportButton = page.locator('button:has-text("Export Trade Log")');
    await expect(exportButton).toBeVisible();
  });
});
