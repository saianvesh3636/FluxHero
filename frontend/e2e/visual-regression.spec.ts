import { test, expect } from '@playwright/test';

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

    // Take full page screenshot
    await expect(page).toHaveScreenshot('home-page.png', {
      fullPage: true,
    });
  });

  test('live trading page visual snapshot', async ({ page }) => {
    await page.goto('/live');
    await page.waitForLoadState('networkidle');

    // Wait for loading to complete
    await page.waitForSelector('text=Loading live data...', {
      state: 'hidden',
      timeout: 10000
    }).catch(() => {
      // If loading spinner doesn't exist, that's fine
    });

    // Wait for any animations to complete
    await page.waitForTimeout(1000);

    // Take full page screenshot
    await expect(page).toHaveScreenshot('live-trading-page.png', {
      fullPage: true,
    });
  });

  test('analytics page visual snapshot', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');

    // Wait for charts to render
    await page.waitForTimeout(2000);

    // Take full page screenshot
    await expect(page).toHaveScreenshot('analytics-page.png', {
      fullPage: true,
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
    });
  });

  test.describe('Component-level visual tests', () => {
    test('positions table component', async ({ page }) => {
      await page.goto('/live');
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('text=Loading live data...', {
        state: 'hidden',
        timeout: 10000
      }).catch(() => {});

      // Locate the positions section
      const positionsSection = page.locator('h2:has-text("Open Positions")').locator('..');

      if (await positionsSection.isVisible()) {
        await expect(positionsSection).toHaveScreenshot('positions-table.png');
      }
    });

    test('account summary component', async ({ page }) => {
      await page.goto('/live');
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('text=Loading live data...', {
        state: 'hidden',
        timeout: 10000
      }).catch(() => {});

      // Locate the stats grid
      const statsGrid = page.locator('.stats-grid').first();

      if (await statsGrid.isVisible()) {
        await expect(statsGrid).toHaveScreenshot('account-summary.png');
      }
    });

    test('system status indicator', async ({ page }) => {
      await page.goto('/live');
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('text=Loading live data...', {
        state: 'hidden',
        timeout: 10000
      }).catch(() => {});

      // Locate system status card
      const statusCard = page.locator('text=/System Status/i').locator('..');

      if (await statusCard.isVisible()) {
        await expect(statusCard).toHaveScreenshot('system-status.png');
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
      });
    });

    test('tablet viewport - live page', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.goto('/live');
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('text=Loading live data...', {
        state: 'hidden',
        timeout: 10000
      }).catch(() => {});
      await page.waitForTimeout(1000);

      await expect(page).toHaveScreenshot('live-tablet.png', {
        fullPage: true,
      });
    });

    test('desktop viewport - analytics page', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 });
      await page.goto('/analytics');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(2000);

      await expect(page).toHaveScreenshot('analytics-desktop.png', {
        fullPage: true,
      });
    });
  });

  test.describe('Dark mode visual tests', () => {
    test('home page - dark mode', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      // Toggle dark mode if toggle exists
      const darkModeToggle = page.locator('button:has-text("Dark"), button:has-text("Light")');
      if (await darkModeToggle.isVisible().catch(() => false)) {
        await darkModeToggle.click();
        await page.waitForTimeout(500);

        await expect(page).toHaveScreenshot('home-dark-mode.png', {
          fullPage: true,
        });
      }
    });

    test('live page - dark mode', async ({ page }) => {
      await page.goto('/live');
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('text=Loading live data...', {
        state: 'hidden',
        timeout: 10000
      }).catch(() => {});

      // Toggle dark mode if toggle exists
      const darkModeToggle = page.locator('button:has-text("Dark"), button:has-text("Light")');
      if (await darkModeToggle.isVisible().catch(() => false)) {
        await darkModeToggle.click();
        await page.waitForTimeout(1000);

        await expect(page).toHaveScreenshot('live-dark-mode.png', {
          fullPage: true,
        });
      }
    });
  });

  test.describe('Error state visual tests', () => {
    test('error message display', async ({ page }) => {
      // Test error state by navigating to a page and simulating offline backend
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

      // Capture loading state immediately
      await page.waitForTimeout(500);

      const loadingSpinner = page.locator('text=Loading');
      if (await loadingSpinner.isVisible().catch(() => false)) {
        await expect(page).toHaveScreenshot('loading-state.png');
      }
    });
  });
});
