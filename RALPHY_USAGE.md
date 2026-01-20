# Using Ralphy with FluxHero

This guide explains how to use Ralphy to autonomously implement FluxHero features.

## What is Ralphy?

Ralphy is an AI agent orchestration tool that automates coding tasks. It spins up Claude Code agents to work through your task list autonomously, creating branches, running tests, and committing changes.

## Quick Start

### 1. Check Setup
Ralphy is already configured for FluxHero:
- ✅ Ralphy cloned to `/Users/anvesh/Developer/QuantTrading/ralphy/`
- ✅ Configuration at `.ralphy/config.yaml`
- ✅ Task list at `FLUXHERO_TASKS.md` (118 tasks)
- ✅ Helper script at `run-ralphy.sh`

### 2. Run Ralphy

**Option A: Using the helper script (Recommended)**
```bash
./run-ralphy.sh
```

**Option B: Manual command**
```bash
/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh \
  --prd FLUXHERO_TASKS.md \
  --branch-per-task
```

**Option C: Single task test**
```bash
/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh "Create project folder structure"
```

## Execution Configuration

Based on your preferences, Ralphy is configured for:

| Setting | Value | Why |
|---------|-------|-----|
| **Execution mode** | Sequential | One task at a time for safer, easier-to-debug implementation |
| **Branching** | Branch per task | Each task gets its own git branch for clean isolation |
| **Pull Requests** | Manual | No auto PRs - you review and merge branches manually |
| **AI Engine** | Claude Code | Default engine (the same AI you're talking to now) |
| **Task scope** | Full system | All 118 tasks across 12 features |

## What Ralphy Does

For each task in `FLUXHERO_TASKS.md`, Ralphy will:

1. ✅ Create a new git branch (e.g., `task-1-create-project-structure`)
2. ✅ Spin up a Claude Code agent with your project context
3. ✅ Execute the task (write code, create files, run tests)
4. ✅ Commit changes with descriptive message
5. ✅ Move to next task (or stop if error)

## Monitoring Progress

### Real-time Progress
```bash
tail -f .ralphy/progress.txt
```

### Check Configuration
```bash
/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh --config
```

### View Task List
```bash
cat FLUXHERO_TASKS.md
```

## Task List Structure

The task list has 18 phases:

| Phase | Tasks | Description |
|-------|-------|-------------|
| Phase 1 | 4 tasks | Project setup & infrastructure |
| Phase 2 | 6 tasks | JIT computation engine (Numba) |
| Phase 3 | 6 tasks | Adaptive EMA (KAMA) |
| Phase 4 | 6 tasks | Volatility-adaptive smoothing |
| Phase 5 | 6 tasks | Market noise filter |
| Phase 6 | 7 tasks | Regime detection |
| Phase 7 | 8 tasks | Dual-mode strategy engine |
| Phase 8 | 7 tasks | Lightweight storage (SQLite/Parquet) |
| Phase 9 | 8 tasks | Async API wrapper |
| Phase 10 | 9 tasks | Backtesting module |
| Phase 11 | 8 tasks | Order execution engine |
| Phase 12 | 7 tasks | Risk management system |
| Phase 13 | 8 tasks | Backend API (FastAPI) |
| Phase 14 | 17 tasks | Frontend dashboard (React/Next.js) |
| Phase 15 | 5 tasks | Signal explainer & logging |
| Phase 16 | 4 tasks | Retail-specific optimizations |
| Phase 17 | 7 tasks | Testing & validation |
| Phase 18 | 6 tasks | Documentation & deployment |
| **Total** | **118 tasks** | Full FluxHero implementation |

## Manual Task Execution

If you want to run specific tasks manually instead of the full loop:

### Execute a single task
```bash
/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh "Implement backend/computation/indicators.py with Numba @njit decorator"
```

### Execute a specific phase
1. Copy tasks from one phase into a temporary file (e.g., `phase1.md`)
2. Run: `/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh --prd phase1.md`

## Stopping Ralphy

- Press `Ctrl+C` to stop after current task completes
- Ralphy saves progress in `.ralphy/progress.txt`
- You can resume later from where you left off

## Reviewing Changes

After Ralphy runs tasks:

1. **Check branches created:**
   ```bash
   git branch -a | grep task-
   ```

2. **Review a specific task branch:**
   ```bash
   git checkout task-1-create-project-structure
   git log -1 --stat
   git diff main
   ```

3. **Merge approved tasks:**
   ```bash
   git checkout main
   git merge task-1-create-project-structure
   ```

4. **Delete merged branch:**
   ```bash
   git branch -d task-1-create-project-structure
   ```

## Advanced Options

### Run in parallel (faster, but more complex)
```bash
/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh \
  --prd FLUXHERO_TASKS.md \
  --parallel \
  --max-parallel 3 \
  --branch-per-task
```

### Dry run (preview what would be done)
```bash
/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh \
  --prd FLUXHERO_TASKS.md \
  --dry-run
```

### Set maximum retries
```bash
/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh \
  --prd FLUXHERO_TASKS.md \
  --branch-per-task \
  --max-retries 5
```

## Customization

### Add new rules
```bash
/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh --add-rule "Your new rule here"
```

### Edit configuration manually
```bash
nano .ralphy/config.yaml
```

### Modify task list
```bash
nano FLUXHERO_TASKS.md
```

## Troubleshooting

### Issue: "Command not found"
**Solution**: Use full path to ralphy.sh:
```bash
/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh --help
```

### Issue: Task fails repeatedly
**Solution**: Check `.ralphy/progress.txt` for error details, fix manually, then resume

### Issue: Git conflicts
**Solution**: Resolve conflicts manually, commit, then resume Ralphy

### Issue: Want to skip a task
**Solution**: Comment out the task in `FLUXHERO_TASKS.md` (prefix with `# `)

## Project Rules (Auto-Applied)

These rules are automatically injected into every Ralphy agent:

- ✅ Use Numba @njit for performance-critical loops
- ✅ Explicit type annotations for Numba compatibility
- ✅ All indicators must run in <100ms for 10k candles
- ✅ Async operations for API calls (httpx, aiohttp)
- ✅ Modular architecture (separate folders per concern)
- ✅ Unit tests required for every feature
- ✅ Backend: Python 3.10+, Frontend: TypeScript strict mode
- ✅ Lightweight storage: SQLite + Parquet (no heavy databases)
- ✅ All signals must include explanation logging
- ✅ Risk management mandatory (1% max risk, stop-loss required)
- ✅ Next-bar fill logic in backtesting
- ✅ Real-time frontend updates via WebSocket

## Files Protected from Modification

Ralphy agents will **never** modify these files:

- `FLUXHERO_REQUIREMENTS.md`
- `FLUXHERO_TASKS.md`
- `README.md`
- `algorithmic-trading-guide.md`
- `quant_trading_guide.md`
- `*.lock` files
- `.ralphy/**` directory
- `data/**/*.db` (SQLite databases)
- `data/**/*.parquet` (cached data)

---

## Ready to Start?

Run the helper script:
```bash
./run-ralphy.sh
```

Or start with a single test task:
```bash
/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh "Create project folder structure for FluxHero"
```

Monitor progress:
```bash
tail -f .ralphy/progress.txt
```

Happy coding! 🚀
