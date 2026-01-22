# FluxHero Parallel Test Execution Tasks

> **Quick Summary**
> - **Total Tasks**: 9 across 3 phases
> - **Focus**: Adding pytest-xdist for parallel test execution
> - **Starting Point**: Phase 1 - Setup & Configuration
> - **Key Deliverable**: Test suite running in parallel with `pytest -n auto`

Configure pytest-xdist for parallel test execution to reduce test feedback time during development.

**Execution Mode**: Sequential
**Source**: Requirements from 2026-01-22

---

## Reference Documents

- REQUIREMENTS.md
- docs/MAINTENANCE_GUIDE.md
- pyproject.toml (or pytest.ini if exists)

---

## Phase 1: Setup & Configuration

- [x] Add `pytest-xdist` to test dependencies in `pyproject.toml` or `requirements-dev.txt`
- [x] Verify installation by running `pytest -n auto` manually and confirming worker processes spawn
- [x] Add `addopts = "-n auto"` to `[tool.pytest.ini_options]` section in `pyproject.toml`
- [x] Verify bare `pytest` command now uses parallel execution by default
- [x] Confirm `pytest -n 0` successfully disables parallel execution when needed

---

## Phase 2: Fix Parallel-Incompatible Tests

- [x] Run full test suite with `pytest -n auto` and document any failing tests
- [x] Fix tests using shared files by switching to `tmp_path` fixture or unique worker-specific paths
- [x] Fix tests binding to specific ports by implementing dynamic port allocation (no issues found - all tests use TestClient)
- [ ] Fix database tests with transaction isolation or worker-specific test databases

---

## Phase 3: Optional Enhancements

- [ ] Add `@pytest.mark.serial` marker configuration if any tests cannot be parallelized (only if needed after Phase 2)
- [ ] Add `make test-fast` target to Makefile for explicit parallel execution (only if team requests)

---

## Notes

- **Fix on demand**: Don't pre-audit tests for parallelism issues—run parallel first, fix failures as they appear
- **Memory monitoring**: Watch memory usage during initial parallel runs, especially if machine has limited RAM
- **Database decision**: If database test failures occur, try transaction rollback isolation first before considering separate databases per worker
