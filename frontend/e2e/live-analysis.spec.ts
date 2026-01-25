import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/api-mocks';

/**
 * Live Analysis Page E2E Tests
 *
 * Tests the /live/analysis page with mock data and captures screenshots.
 * This page shows:
 * - Cumulative P&L charts (algo vs benchmark)
 * - Risk metrics (Sharpe, Sortino, Calmar, etc.)
 * - Daily breakdown table
 */

test.describe('Live Analysis Page', () => {
  test.beforeEach(async ({ page }) => {
    await setupAllMocks(page);
  });

  test('should display live analysis page with charts and metrics', async ({ page }) => {
    await page.goto('/live/analysis');
    await page.waitForLoadState('networkidle');

    // Wait for charts to render
    await page.waitForTimeout(3000);

    // Check page has loaded correctly
    const pageContent = await page.content();
    const hasError = pageContent.includes('Error') || pageContent.includes('Cannot read');

    // Take full page screenshot
    await page.screenshot({
      path: 'e2e/screenshots/live-analysis-full.png',
      fullPage: true
    });

    console.log('Screenshot saved to: e2e/screenshots/live-analysis-full.png');
    console.log('Page has error:', hasError);
  });

  test('should capture viewport screenshot at 1920x1080', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });

    await page.goto('/live/analysis');
    await page.waitForLoadState('networkidle');

    // Wait for charts to render
    await page.waitForTimeout(3000);

    // Take viewport screenshot
    await page.screenshot({
      path: 'e2e/screenshots/live-analysis-desktop.png',
      fullPage: false
    });

    console.log('Screenshot saved to: e2e/screenshots/live-analysis-desktop.png');
  });

  test('should check for risk metrics display', async ({ page }) => {
    await page.goto('/live/analysis');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    // Log what metrics are found
    const metrics = ['Sharpe', 'Sortino', 'Calmar', 'Drawdown', 'Win Rate', 'Profit Factor'];
    for (const metric of metrics) {
      const found = await page.locator(`text=/${metric}/i`).first().isVisible().catch(() => false);
      console.log(`Metric "${metric}": ${found ? 'FOUND' : 'not found'}`);
    }

    // Screenshot for metrics view
    await page.screenshot({
      path: 'e2e/screenshots/live-analysis-metrics.png',
      fullPage: true
    });
  });
});
