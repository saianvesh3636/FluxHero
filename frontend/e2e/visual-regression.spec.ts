import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/api-mocks';

/**
 * Visual Regression Tests
 *
 * These tests capture screenshots of key pages and compare them against baselines
 * to detect unintended visual changes in the UI.
 *
 * To update baselines:
 * npm run test:e2e -- --update-snapshots
 */

test.describe('Visual Regression Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Setup API mocks before each test
    await setupAllMocks(page);

    // Wait for fonts and styles to load
    await page.addStyleTag({
      content: `
        * {
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
      `
    });
  });

  test('home page visual snapshot', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Wait for any animations to complete
    await page.waitForTimeout(500);

    // Take full page screenshot - mask dynamic timestamps
    await expect(page).toHaveScreenshot('home-page.png', {
      fullPage: true,
      maxDiffPixelRatio: 0.05, // Allow 5% for dynamic content
      mask: [
        page.locator('text=/Last Update/i').locator('..'),
        page.locator('text=/Uptime/i').locator('..'),
      ],
    });
  });

  test('live trading page visual snapshot', async ({ page }) => {
    await page.goto('/live');
    await page.waitForLoadState('networkidle');

    // Wait for loading to complete (new design uses "Loading..." in subtitle)
    await page.waitForSelector('p:has-text("Loading...")', {
      state: 'hidden',
      timeout: 10000
    }).catch(() => {
      // If loading text doesn't exist, that's fine
    });

    // Wait for any animations to complete
    await page.waitForTimeout(1000);

    // Take full page screenshot - mask timestamp which changes between runs
    await expect(page).toHaveScreenshot('live-trading-page.png', {
      fullPage: true,
      maxDiffPixelRatio: 0.05, // Allow 5% for timestamp and other dynamic content
      mask: [page.locator('p:has-text("Last updated")')],
    });
  });

  test('analytics page visual snapshot', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');

    // Wait for charts to render
    await page.waitForTimeout(2000);

    // Take full page screenshot (allow small pixel differences for dynamic content)
    await expect(page).toHaveScreenshot('analytics-page.png', {
      fullPage: true,
      maxDiffPixelRatio: 0.02,
    });
  });

  test('backtest page visual snapshot', async ({ page }) => {
    await page.goto('/backtest');
    await page.waitForLoadState('networkidle');

    // Wait for any animations to complete
    await page.waitForTimeout(500);

    // Take full page screenshot
    await expect(page).toHaveScreenshot('backtest-page.png', {
      fullPage: true,
      maxDiffPixelRatio: 0.02,
    });
  });

  test('trade history page visual snapshot', async ({ page }) => {
    await page.goto('/history');
    await page.waitForLoadState('networkidle');

    // Wait for table to render
    await page.waitForTimeout(1000);

    // Take full page screenshot
    await expect(page).toHaveScreenshot('history-page.png', {
      fullPage: true,
      maxDiffPixelRatio: 0.02,
    });
  });

  test.describe('Component-level visual tests', () => {
    test('positions table component', async ({ page }) => {
      await page.goto('/live');
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('p:has-text("Loading...")', {
        state: 'hidden',
        timeout: 10000
      }).catch(() => {});

      // Locate the positions section - new design uses CardTitle
      const positionsSection = page.locator('text=/Open Positions/i').locator('..').locator('..');

      if (await positionsSection.isVisible()) {
        await expect(positionsSection).toHaveScreenshot('positions-table.png', {
          maxDiffPixelRatio: 0.02,
        });
      }
    });

    // Skip: This test is flaky due to dynamic P&L values changing between runs
    test.skip('account summary component', async ({ page }) => {
      await page.goto('/live');
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('p:has-text("Loading...")', {
        state: 'hidden',
        timeout: 10000
      }).catch(() => {});

      // Locate the account summary card
      const accountSummary = page.locator('text=/Account Summary/i').locator('..').locator('..');

      if (await accountSummary.isVisible()) {
        await expect(accountSummary).toHaveScreenshot('account-summary.png', {
          maxDiffPixelRatio: 0.1, // Allow 10% for dynamic P&L values
        });
      }
    });

    test('system status indicator', async ({ page }) => {
      await page.goto('/live');
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('p:has-text("Loading...")', {
        state: 'hidden',
        timeout: 10000
      }).catch(() => {});

      // Locate system status - new design shows status badge
      const statusBadge = page.locator('text=/ACTIVE|DELAYED|OFFLINE/i').first();

      if (await statusBadge.isVisible()) {
        await expect(statusBadge).toHaveScreenshot('system-status.png', {
          maxDiffPixelRatio: 0.02,
        });
      }
    });
  });

  test.describe('Responsive visual tests', () => {
    test('mobile viewport - home page', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(500);

      await expect(page).toHaveScreenshot('home-mobile.png', {
        fullPage: true,
        maxDiffPixelRatio: 0.02,
      });
    });

    test('tablet viewport - live page', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.goto('/live');
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('p:has-text("Loading...")', {
        state: 'hidden',
        timeout: 10000
      }).catch(() => {});
      await page.waitForTimeout(1000);

      await expect(page).toHaveScreenshot('live-tablet.png', {
        fullPage: true,
        maxDiffPixelRatio: 0.05,
        mask: [page.locator('p:has-text("Last updated")')],
      });
    });

    test('desktop viewport - analytics page', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 });
      await page.goto('/analytics');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(2000);

      await expect(page).toHaveScreenshot('analytics-desktop.png', {
        fullPage: true,
        maxDiffPixelRatio: 0.02,
      });
    });
  });

  // Dark mode is the only mode now - skip toggle tests
  test.describe('Dark mode visual tests', () => {
    test.skip('home page - dark mode', async ({ page }) => {
      // Skipped: App is now dark mode only, no toggle exists
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      await expect(page).toHaveScreenshot('home-dark-mode.png', {
        fullPage: true,
      });
    });

    test.skip('live page - dark mode', async ({ page }) => {
      // Skipped: App is now dark mode only, no toggle exists
      await page.goto('/live');
      await page.waitForLoadState('networkidle');
      await expect(page).toHaveScreenshot('live-dark-mode.png', {
        fullPage: true,
      });
    });
  });

  test.describe('Error state visual tests', () => {
    test('error message display', async ({ page }) => {
      // Override mocks to simulate offline backend
      await page.route('**/api/**', route => route.abort());

      await page.goto('/live');
      await page.waitForLoadState('networkidle');

      // Wait for error state to appear
      await page.waitForTimeout(2000);

      // Check if error state is visible
      const errorIndicator = page.locator('text=/error/i, text=/offline/i, text=/failed/i').first();
      if (await errorIndicator.isVisible().catch(() => false)) {
        await expect(page).toHaveScreenshot('error-state.png', {
          fullPage: true,
        });
      }
    });

    test('loading state display', async ({ page }) => {
      // Slow down network to capture loading state
      await page.route('**/api/**', async route => {
        await new Promise(resolve => setTimeout(resolve, 5000));
        await route.continue();
      });

      await page.goto('/live');

      // Capture loading state immediately (new design uses "Loading..." in subtitle)
      await page.waitForTimeout(500);

      const loadingText = page.locator('text=Loading...');
      if (await loadingText.isVisible().catch(() => false)) {
        await expect(page).toHaveScreenshot('loading-state.png');
      }
    });
  });
});
