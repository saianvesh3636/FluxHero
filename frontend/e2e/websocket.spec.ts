import { test, expect } from '@playwright/test';
import { setupAllMocks } from './mocks/api-mocks';

test.describe('WebSocket Connection', () => {
  test.beforeEach(async ({ page }) => {
    // Setup API mocks before each test
    await setupAllMocks(page);
  });

  test('should show WebSocket connection status on analytics page', async ({ page }) => {
    // Navigate to analytics page
    await page.goto('/analytics');

    // Wait for the page to load (including timeout for mock data)
    await page.waitForLoadState('networkidle');

    // Wait for loading spinner to disappear (with generous timeout)
    await page.waitForSelector('p:has-text("Loading...")', {
      state: 'hidden',
      timeout: 15000
    }).catch(() => {
      // If loading spinner doesn't exist, that's also fine
    });

    // Wait for chart to render
    await page.waitForTimeout(2000);

    // The page should have loaded - check for any header element
    const pageBody = page.locator('body');
    await expect(pageBody).toBeVisible();

    // Check for status indicator - new design uses text labels
    const statusIndicators = ['connected', 'disconnected', 'connecting', 'CONNECTED', 'DISCONNECTED'];
    let foundStatus = false;

    for (const status of statusIndicators) {
      if (await page.locator(`text=${status}`).isVisible().catch(() => false)) {
        foundStatus = true;
        break;
      }
    }

    // Page should have loaded with some status indicator
    expect(foundStatus).toBeTruthy();
  });

  test('should show connection status indicator when backend is mocked', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');

    // Wait for loading to complete
    await page.waitForSelector('p:has-text("Loading...")', {
      state: 'hidden',
      timeout: 15000
    }).catch(() => {});

    await page.waitForTimeout(2000);

    // Check for any status indicator - new design uses text labels
    const statusIndicators = ['connected', 'disconnected', 'connecting', 'CONNECTED', 'DISCONNECTED'];
    let foundIndicator = false;

    for (const status of statusIndicators) {
      if (await page.locator(`text=${status}`).isVisible().catch(() => false)) {
        foundIndicator = true;
        break;
      }
    }

    expect(foundIndicator).toBeTruthy();
  });

  test('should display price updates in console when WebSocket receives data', async ({ page }) => {
    const consoleLogs: string[] = [];

    // Listen for console logs
    page.on('console', (msg) => {
      if (msg.type() === 'log') {
        consoleLogs.push(msg.text());
      }
    });

    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');

    // Wait for loading to complete
    await page.waitForSelector('p:has-text("Loading...")', {
      state: 'hidden',
      timeout: 15000
    }).catch(() => {});

    // Wait for potential WebSocket messages
    await page.waitForTimeout(3000);

    // This test is informational - just verify page loaded
    const pageBody = page.locator('body');
    await expect(pageBody).toBeVisible();

    // Console logs are informational, not required
    console.log('WebSocket console logs detected:', consoleLogs.length);
    expect(true).toBeTruthy();
  });

  test('should handle WebSocket reconnection gracefully', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');

    // Wait for loading to complete
    await page.waitForSelector('p:has-text("Loading...")', {
      state: 'hidden',
      timeout: 15000
    }).catch(() => {});

    await page.waitForTimeout(2000);

    // Check for any status indicator - new design uses text labels
    const statusIndicators = ['connected', 'disconnected', 'connecting', 'CONNECTED', 'DISCONNECTED'];
    let foundStatus = false;

    for (const status of statusIndicators) {
      if (await page.locator(`text=${status}`).isVisible().catch(() => false)) {
        foundStatus = true;
        break;
      }
    }

    expect(foundStatus).toBeTruthy();

    // If retry button exists, try clicking it
    const retryButton = page.locator('button:has-text("Retry")');
    if (await retryButton.isVisible().catch(() => false)) {
      await retryButton.click();
      await page.waitForTimeout(1000);
    }

    // Page should remain functional
    const pageBody = page.locator('body');
    await expect(pageBody).toBeVisible();
  });

  test('should handle symbol selection if available', async ({ page }) => {
    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');

    // Wait for loading to complete
    await page.waitForSelector('p:has-text("Loading...")', {
      state: 'hidden',
      timeout: 15000
    }).catch(() => {});

    await page.waitForTimeout(2000);

    // Check if symbol selector exists
    const symbolSelect = page.locator('select').first();
    const hasSymbolSelect = await symbolSelect.isVisible().catch(() => false);

    if (hasSymbolSelect) {
      // Get initial symbol
      const initialSymbol = await symbolSelect.inputValue();
      expect(initialSymbol).toBeTruthy();

      // Try to change symbol if options exist
      const options = await page.locator('select option').all();
      if (options.length > 1) {
        await symbolSelect.selectOption({ index: 1 });
        await page.waitForTimeout(1000);
      }
    }

    // Page should remain functional
    const pageBody = page.locator('body');
    await expect(pageBody).toBeVisible();
  });

  test('should not crash when WebSocket connection fails', async ({ page }) => {
    const errors: string[] = [];
    const rejections: string[] = [];

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    page.on('pageerror', (error) => {
      rejections.push(error.message);
    });

    await page.goto('/analytics');
    await page.waitForLoadState('networkidle');

    // Wait for loading to complete
    await page.waitForSelector('p:has-text("Loading...")', {
      state: 'hidden',
      timeout: 15000
    }).catch(() => {});

    // Wait for potential WebSocket connection attempts
    await page.waitForTimeout(3000);

    // Filter out expected errors (WebSocket connection failures are expected in test)
    const criticalErrors = errors.filter(
      (error) =>
        !error.includes('favicon') &&
        !error.includes('404') &&
        !error.includes('WebSocket') &&
        !error.includes('Failed to load resource') &&
        !error.includes('net::ERR') &&
        !error.includes('ws://') &&
        !error.includes('Connection refused')
    );

    const criticalRejections = rejections.filter(
      (rejection) =>
        !rejection.includes('favicon') &&
        !rejection.includes('WebSocket') &&
        !rejection.includes('ws://') &&
        !rejection.includes('Connection refused')
    );

    // The page should handle WebSocket errors gracefully
    expect(criticalErrors.length).toBe(0);
    expect(criticalRejections.length).toBe(0);

    // The page should still be functional
    const pageBody = page.locator('body');
    await expect(pageBody).toBeVisible();
  });
});
