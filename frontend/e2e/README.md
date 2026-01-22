# E2E Tests

This directory contains end-to-end tests for the FluxHero frontend using Playwright.

## Test Files

- `home.spec.ts` - Home page functionality tests
- `live.spec.ts` - Live trading page tests
- `backtest.spec.ts` - Backtest page tests
- `error-states.spec.ts` - Error handling tests
- `websocket.spec.ts` - WebSocket connection tests
- `visual-regression.spec.ts` - Visual regression tests with screenshots

## Running Tests

### Run all tests
```bash
npm run test:e2e
```

### Run tests in UI mode (interactive)
```bash
npx playwright test --ui
```

### Run specific test file
```bash
npx playwright test e2e/visual-regression.spec.ts
```

### Run tests in headed mode (see browser)
```bash
npx playwright test --headed
```

## Visual Regression Testing

Visual regression tests capture screenshots of the UI and compare them against baseline images to detect unintended visual changes.

### Initial Setup - Generate Baseline Screenshots

The first time you run visual regression tests, you need to generate baseline screenshots:

```bash
npx playwright test e2e/visual-regression.spec.ts
```

This will create baseline screenshots in `e2e/visual-regression.spec.ts-snapshots/` directory.

### Updating Baselines

If you intentionally change the UI, update the baseline screenshots:

```bash
npx playwright test e2e/visual-regression.spec.ts --update-snapshots
```

Or update all snapshots:

```bash
npm run test:e2e -- --update-snapshots
```

### Review Visual Differences

When a visual regression test fails, Playwright generates:
- **Actual screenshot** - What the current UI looks like
- **Expected screenshot** - The baseline
- **Diff screenshot** - Highlighted differences

View the HTML report to see visual differences:

```bash
npx playwright show-report
```

## Test Categories

### 1. Full Page Snapshots
- Home page
- Live trading page
- Analytics page
- Backtest page
- Trade history page

### 2. Component-Level Snapshots
- Positions table
- Account summary
- System status indicator

### 3. Responsive Snapshots
- Mobile viewport (375x667)
- Tablet viewport (768x1024)
- Desktop viewport (1920x1080)

### 4. Dark Mode Snapshots
- Home page in dark mode
- Live page in dark mode

### 5. State-Based Snapshots
- Error states
- Loading states

## Configuration

Visual regression settings are configured in `playwright.config.ts`:

```typescript
expect: {
  toHaveScreenshot: {
    maxDiffPixels: 100,      // Allow up to 100 pixels to differ
    threshold: 0.2,          // 20% pixel difference threshold
  },
},
```

### Adjusting Tolerance

If tests are too sensitive (failing on minor changes):
- Increase `maxDiffPixels` (e.g., 200)
- Increase `threshold` (e.g., 0.3 for 30%)

If tests are too lenient (missing real changes):
- Decrease `maxDiffPixels` (e.g., 50)
- Decrease `threshold` (e.g., 0.1 for 10%)

## Best Practices

### 1. Stable Rendering
- Wait for `networkidle` before taking screenshots
- Add timeouts to let animations complete
- Use anti-aliasing CSS for consistent font rendering

### 2. Dynamic Content
- Mock API responses for consistent data
- Avoid time-based content (use fixed dates in tests)
- Consider masking dynamic regions

### 3. CI/CD Integration
- Store baseline screenshots in version control
- Update baselines deliberately, not automatically
- Review visual diffs before approving changes

### 4. Maintenance
- Review and update baselines when UI changes are intentional
- Keep snapshots directory clean (remove obsolete screenshots)
- Document major UI changes that affect baselines

## Troubleshooting

### Tests fail with visual differences

1. Check the HTML report: `npx playwright show-report`
2. Review the diff screenshots to identify changes
3. If changes are intentional, update baselines: `--update-snapshots`
4. If changes are bugs, fix the UI

### Inconsistent results across environments

- Ensure all developers use the same OS (or use Docker)
- Font rendering can differ between OS - consider using web fonts
- Use the same Playwright version across team

### Flaky visual tests

- Increase wait times for dynamic content
- Disable animations in tests
- Mock time-dependent data
- Use `{ animations: 'disabled' }` in page options

## Example Workflow

1. **Develop new feature**
   ```bash
   # Make UI changes
   ```

2. **Run tests locally**
   ```bash
   npm run test:e2e
   ```

3. **Review visual differences**
   ```bash
   npx playwright show-report
   ```

4. **Update baselines if intentional**
   ```bash
   npx playwright test e2e/visual-regression.spec.ts --update-snapshots
   ```

5. **Commit changes including updated baselines**
   ```bash
   git add e2e/visual-regression.spec.ts-snapshots/
   git commit -m "feat: Update UI with new design"
   ```

## CI Configuration

For GitHub Actions or other CI:

```yaml
- name: Run Playwright tests
  run: npm run test:e2e

- name: Upload test results
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: playwright-report
    path: playwright-report/
```

## Resources

- [Playwright Documentation](https://playwright.dev)
- [Visual Comparisons Guide](https://playwright.dev/docs/test-snapshots)
- [Best Practices](https://playwright.dev/docs/best-practices)
