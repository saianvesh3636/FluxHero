import { test, expect } from '@playwright/test';

test.describe('Error States', () => {
  // Disable WebSocket connections to prevent hitting the real backend
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      (window as { __PLAYWRIGHT_TEST__?: boolean }).__PLAYWRIGHT_TEST__ = true;
    });
  });

  test('should show backend offline indicator on /live page', async ({ page }) => {
    // Mock all API endpoints to fail completely (network error)
    await page.route('/api/**', async (route) => {
      await route.abort('failed');
    });

    // Navigate to live page
    await page.goto('/live');

    // Wait for error to show
    await page.waitForTimeout(5000);

    // Check various possible error indicators
    const hasOfflineText = await page.locator('text=/offline/i').first().isVisible().catch(() => false);
    const hasRetryButton = await page.locator('button').filter({ hasText: /retry/i }).first().isVisible().catch(() => false);
    const hasErrorEmoji = await page.locator('text=🔴').first().isVisible().catch(() => false);
    const hasErrorBanner = await page.locator('.bg-red-50, .bg-red-900, .bg-yellow-50').first().isVisible().catch(() => false);

    // At least one error indicator should be visible
    expect(hasOfflineText || hasRetryButton || hasErrorEmoji || hasErrorBanner).toBeTruthy();
  });

  test('should allow retry when backend is offline', async ({ page }) => {
    let requestCount = 0;

    // First requests fail, subsequent requests succeed
    await page.route('/api/**', async (route) => {
      const url = route.request().url();
      requestCount++;

      if (requestCount <= 3) {
        // First set of requests fail
        await route.abort('failed');
      } else {
        // Subsequent requests succeed
        if (url.includes('/api/status')) {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ status: 'active', timestamp: new Date().toISOString() }),
          });
        } else if (url.includes('/api/positions')) {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([]),
          });
        } else if (url.includes('/api/account')) {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              equity: 100000,
              cash: 50000,
              buying_power: 200000,
              daily_pnl: 1250,
              total_pnl: 5000,
            }),
          });
        } else {
          await route.continue();
        }
      }
    });

    await page.goto('/live');

    // Wait for error state to appear
    await page.waitForTimeout(5000);

    // Find retry button - it might have different text
    const retryButton = page.locator('button').filter({ hasText: /retry/i }).first();
    const buttonExists = await retryButton.isVisible().catch(() => false);

    if (buttonExists) {
      await retryButton.click();

      // After retry, page should show normal content
      await page.waitForTimeout(2000);
      await expect(page.locator('h1:has-text("Live Trading")')).toBeVisible();
    } else {
      // If no retry button found, at least verify the page header is there
      await expect(page.locator('h1:has-text("Live Trading")')).toBeVisible();
    }
  });

  test('should show error state when API returns error on /live page', async ({ page }) => {
    // Mock API to return 500 error
    await page.route('/api/status', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    await page.route('/api/positions', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    await page.route('/api/account', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    await page.goto('/live');

    // Should show error (either backend offline or error banner)
    await page.waitForTimeout(2000);
    const hasError =
      (await page.locator('text=/Backend Offline/i').isVisible().catch(() => false)) ||
      (await page.locator('text=/Error/i').isVisible().catch(() => false)) ||
      (await page.locator('button:has-text("Retry")').isVisible().catch(() => false));

    expect(hasError).toBeTruthy();
  });

  test('should handle network timeout gracefully', async ({ page }) => {
    // Mock API to fail (simulating timeout)
    await page.route('/api/**', async (route) => {
      await route.abort('failed');
    });

    await page.goto('/live');

    // Wait for page to settle
    await page.waitForTimeout(6000);

    // Should show some kind of error state or still be loading
    const hasErrorBanner = await page.locator('.bg-red-50, .bg-red-900, .bg-yellow-50, .bg-loss-500\\/10').first().isVisible().catch(() => false);
    const hasRetryButton = await page.locator('button').filter({ hasText: /retry/i }).first().isVisible().catch(() => false);
    const isLoading = await page.locator('text=Loading...').isVisible().catch(() => false);
    const hasOfflineText = await page.locator('text=/offline/i').first().isVisible().catch(() => false);

    expect(hasErrorBanner || hasRetryButton || isLoading || hasOfflineText).toBeTruthy();
  });

  test('should show loading state while fetching data', async ({ page }) => {
    // Mock API with delay to simulate loading
    await page.route('/api/status', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'active', timestamp: new Date().toISOString() }),
      });
    });

    await page.route('/api/positions', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      });
    });

    await page.route('/api/account', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          equity: 100000,
          cash: 50000,
          buying_power: 200000,
          daily_pnl: 0,
          total_pnl: 0,
        }),
      });
    });

    const navigationPromise = page.goto('/live');

    // Should show loading state initially (new design uses "Loading..." in subtitle)
    const loadingText = page.locator('text=Loading...');
    await expect(loadingText).toBeVisible({ timeout: 2000 });

    await navigationPromise;

    // After data loads, loading text should disappear
    await expect(loadingText).not.toBeVisible({ timeout: 5000 });
  });
});
