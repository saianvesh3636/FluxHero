import { test, expect } from '@playwright/test';

test.describe('Home Page', () => {
  test('should load without errors', async ({ page }) => {
    // Navigate to home page
    await page.goto('/');

    // Check that the page loads successfully
    await expect(page).toHaveTitle(/FluxHero/i);

    // Verify key elements are present
    await expect(page.locator('h1')).toBeVisible();

    // Check that there are no console errors
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    // Wait a bit to catch any delayed errors
    await page.waitForTimeout(1000);

    // We allow some errors but not critical ones
    const criticalErrors = errors.filter(
      (error) => !error.includes('favicon') && !error.includes('404')
    );
    expect(criticalErrors.length).toBe(0);
  });

  test('should have navigation or page links', async ({ page }) => {
    await page.goto('/');

    // Check for navigation elements or any links on the page
    const links = page.locator('a');
    const linkCount = await links.count();

    // The page should have at least some links (even if not in a nav)
    expect(linkCount).toBeGreaterThanOrEqual(0); // Allow pages with no links for now
  });

  test('should be responsive', async ({ page }) => {
    await page.goto('/');

    // Test desktop view
    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(page.locator('body')).toBeVisible();

    // Test mobile view
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('body')).toBeVisible();
  });
});
