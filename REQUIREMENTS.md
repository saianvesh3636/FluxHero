# Parallel Test Execution Requirements

> **Quick Summary**
> - **Building**: Parallel test runner to execute tests concurrently instead of sequentially
> - **For**: Developers on FluxHero project who need faster test feedback during development
> - **Core Features**: Multi-worker test execution, auto-detection of CPU cores, test isolation
> - **Tech**: pytest-xdist plugin
> - **Scope**: Small (configuration change + minimal test fixes)

## Overview

The FluxHero test suite currently runs sequentially, creating a bottleneck during development iteration. By adding pytest-xdist, tests can run across multiple CPU cores simultaneously, significantly reducing feedback time. This is a low-risk change using a mature, well-supported pytest plugin.

## Must Have (MVP)

### Feature 1: Parallel Test Execution
- Description: Run tests concurrently across multiple worker processes using pytest-xdist
- Acceptance Criteria:
  - [x] `pytest-xdist` is added to test dependencies
  - [x] `pytest -n auto` successfully runs the test suite with auto-detected worker count
  - [x] `pytest -n <number>` allows manual specification of worker count
  - [x] All currently passing tests continue to pass in parallel mode
- Technical Notes: Install via `pip install pytest-xdist`. The `-n auto` flag detects available CPU cores automatically.

### Feature 2: Default Parallel Configuration
- Description: Configure pytest to use parallel execution by default
- Acceptance Criteria:
  - [x] `pyproject.toml` or `pytest.ini` includes `addopts = "-n auto"` setting
  - [x] Running bare `pytest` command uses parallel execution
  - [x] Parallel execution can be disabled with `pytest -n 0` when needed
- Technical Notes: Add to `[tool.pytest.ini_options]` section in pyproject.toml.

### Feature 3: Fix Parallel-Incompatible Tests
- Description: Identify and fix tests that fail under parallel execution
- Acceptance Criteria:
  - [ ] All tests pass when run with `pytest -n auto`
  - [ ] Tests using shared files use `tmp_path` fixture or unique paths per worker
  - [ ] Tests binding to specific ports use dynamic port allocation
  - [ ] Database tests use isolated test databases or proper transaction isolation
- Technical Notes: Run parallel tests first, then fix failures as they appear. Don't pre-audit—fix on demand.

## Should Have (Post-MVP)

### Serial Test Marker
- Description: Mark specific tests that must run sequentially
- Acceptance Criteria:
  - [ ] `@pytest.mark.serial` decorator available for tests requiring sequential execution
  - [ ] Serial-marked tests run in a single worker while others parallelize
- Technical Notes: Only add if specific tests cannot be fixed for parallel execution. Use `pytest-xdist`'s `--dist loadfile` as alternative.

### Convenience Make Target
- Description: Add `make test-fast` command for parallel testing
- Acceptance Criteria:
  - [ ] `make test-fast` runs `pytest -n auto`
  - [ ] `make test` remains available for sequential execution if needed
- Technical Notes: Only add if team requests it—`pytest -n auto` is already simple.

## Nice to Have (Future)

- **Test distribution strategies**: Experiment with `--dist loadscope` or `--dist loadfile` for optimizing test distribution
- **CI parallelization**: Configure CI pipeline to leverage parallel testing
- **Memory profiling**: Monitor memory usage with multiple workers in resource-constrained environments

## Out of Scope

- Custom progress reporting (xdist has built-in progress)
- Custom test isolation frameworks (use pytest's existing fixtures)
- Distributed testing across multiple machines
- Test result aggregation dashboards
- Pre-auditing all tests for parallelism issues

## Technical Constraints

- **Memory**: Each worker loads full test environment; monitor memory on machines with limited RAM
- **CPU bound**: Diminishing returns beyond CPU core count; `-n auto` handles this
- **Worker startup overhead**: For very small test suites (<50 tests), parallel overhead may exceed savings
- **Session-scoped fixtures**: May behave unexpectedly—each worker gets its own session

## Open Questions

- **Database test strategy**: If tests share a test database, decide between:
  1. Transaction rollback isolation per test
  2. Separate test database per worker
  3. Mark database tests as serial
  
  *Recommendation: Try running first, fix based on actual failures.*

- **Minimum test count threshold**: Should there be a threshold below which parallel execution is skipped? (Likely not needed—let users decide with `-n 0`)
