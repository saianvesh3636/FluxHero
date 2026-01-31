# FluxHero Development Makefile
# =============================
# Provides unified commands for development workflow
#
# Usage:
#   make help          - Show all available commands
#   make dev           - Start both backend and frontend in development mode
#   make stop          - Stop all running services
#   make test          - Run all tests
#   make lint          - Run linting and type checking

.PHONY: help dev dev-backend dev-debug dev-frontend stop stop-backend stop-frontend \
        test test-unit test-integration test-parallel lint format typecheck \
        install install-backend install-frontend clean logs notebook notebook-lab \
        docker-build docker-up docker-down docker-logs docker-shell-backend docker-clean \
        backtest backtest-optimize backtest-live backtest-quick backtest-compare \
        backtest-full backtest-report backtest-report-full backtest-all backtest-all-quick

# Colors for terminal output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Configuration
VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
BACKEND_PORT := 8000
FRONTEND_PORT := 3000
PID_DIR := .pids

# ============================================================================
# Help
# ============================================================================

help:
	@echo "$(CYAN)FluxHero Development Commands$(NC)"
	@echo "=============================="
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make dev              Start both backend and frontend"
	@echo "  make dev-debug        Start with DEBUG logging"
	@echo "  make dev-backend      Start backend only (port $(BACKEND_PORT))"
	@echo "  make dev-frontend     Start frontend only (port $(FRONTEND_PORT))"
	@echo "  make stop             Stop all running services"
	@echo "  make logs             Show backend logs"
	@echo ""
	@echo "$(GREEN)Backtesting:$(NC)"
	@echo "  make backtest-live           Run backtest with live Yahoo data (SPY, 1yr)"
	@echo "  make backtest-live SYMBOL=X  Backtest custom symbol"
	@echo "  make backtest-quick          Quick 6-month backtest"
	@echo "  make backtest-compare        Compare SPY, QQQ, IWM, DIA"
	@echo "  make backtest-full           Full 2-year analysis"
	@echo "  make backtest-report         Generate HTML tearsheet report"
	@echo "  make backtest-report-full    Full report with Monte Carlo"
	@echo "  make backtest-all            Compare ALL strategies (with report)"
	@echo "  make backtest-all-quick      Quick compare ALL strategies"
	@echo "  make backtest-optimize       Walk-forward grid search"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test             Run all tests (parallel)"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-serial      Run tests serially (no parallel)"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  make lint             Run ruff linting"
	@echo "  make format           Auto-format code with ruff"
	@echo "  make typecheck        Run mypy type checking"
	@echo "  make check            Run all quality checks (lint + typecheck)"
	@echo ""
	@echo "$(GREEN)Notebooks:$(NC)"
	@echo "  make notebook         Start Jupyter Notebook"
	@echo "  make notebook-lab     Start Jupyter Lab"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install          Install all dependencies"
	@echo "  make install-backend  Install Python dependencies"
	@echo "  make install-frontend Install Node.js dependencies"
	@echo "  make clean            Remove generated files and caches"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  make docker-build     Build Docker images"
	@echo "  make docker-up        Start containers in detached mode"
	@echo "  make docker-down      Stop and remove containers"
	@echo "  make docker-logs      Follow container logs"
	@echo "  make docker-shell-backend  Open shell in backend container"
	@echo "  make docker-clean     Remove containers, volumes, and images"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make daily-reboot     Run daily maintenance script"
	@echo "  make archive-trades   Archive old trades to Parquet"
	@echo "  make seed-data        Seed database with test positions"

# ============================================================================
# Development Servers
# ============================================================================

$(PID_DIR):
	@mkdir -p $(PID_DIR)

dev: $(PID_DIR)
	@echo "$(CYAN)Starting FluxHero development servers...$(NC)"
	@$(MAKE) dev-backend &
	@sleep 2
	@$(MAKE) dev-frontend
	@echo "$(GREEN)All services started!$(NC)"
	@echo "  Backend:  http://localhost:$(BACKEND_PORT)"
	@echo "  Frontend: http://localhost:$(FRONTEND_PORT)"
	@echo "  API Docs: http://localhost:$(BACKEND_PORT)/docs"

dev-backend: $(PID_DIR)
	@echo "$(CYAN)Starting backend server on port $(BACKEND_PORT)...$(NC)"
	@if [ -f $(PID_DIR)/backend.pid ] && kill -0 $$(cat $(PID_DIR)/backend.pid) 2>/dev/null; then \
		echo "$(YELLOW)Backend already running (PID: $$(cat $(PID_DIR)/backend.pid))$(NC)"; \
	else \
		$(UVICORN) backend.api.server:app --reload --host 0.0.0.0 --port $(BACKEND_PORT) & \
		echo $$! > $(PID_DIR)/backend.pid; \
		echo "$(GREEN)Backend started (PID: $$!)$(NC)"; \
	fi

dev-debug: $(PID_DIR)
	@echo "$(CYAN)Starting services with DEBUG logging...$(NC)"
	@if [ -f $(PID_DIR)/backend.pid ] && kill -0 $$(cat $(PID_DIR)/backend.pid) 2>/dev/null; then \
		echo "$(YELLOW)Backend already running (PID: $$(cat $(PID_DIR)/backend.pid))$(NC)"; \
	else \
		$(UVICORN) backend.api.server:app --reload --host 0.0.0.0 --port $(BACKEND_PORT) --log-level debug & \
		echo $$! > $(PID_DIR)/backend.pid; \
		echo "$(GREEN)Backend started with DEBUG logging (PID: $$!)$(NC)"; \
	fi
	@$(MAKE) dev-frontend
	@echo "$(GREEN)All services started!$(NC)"
	@echo "  Backend:  http://localhost:$(BACKEND_PORT) (DEBUG)"
	@echo "  Frontend: http://localhost:$(FRONTEND_PORT)"

dev-frontend: $(PID_DIR)
	@echo "$(CYAN)Starting frontend server on port $(FRONTEND_PORT)...$(NC)"
	@if [ -f $(PID_DIR)/frontend.pid ] && kill -0 $$(cat $(PID_DIR)/frontend.pid) 2>/dev/null; then \
		echo "$(YELLOW)Frontend already running (PID: $$(cat $(PID_DIR)/frontend.pid))$(NC)"; \
	else \
		cd frontend && npm run dev & \
		echo $$! > $(PID_DIR)/frontend.pid; \
		echo "$(GREEN)Frontend started (PID: $$!)$(NC)"; \
	fi

stop:
	@echo "$(CYAN)Stopping all services...$(NC)"
	@$(MAKE) stop-backend
	@$(MAKE) stop-frontend
	@echo "$(GREEN)All services stopped$(NC)"

stop-backend:
	@if [ -f $(PID_DIR)/backend.pid ]; then \
		PID=$$(cat $(PID_DIR)/backend.pid); \
		if kill -0 $$PID 2>/dev/null; then \
			kill $$PID; \
			echo "$(GREEN)Backend stopped (PID: $$PID)$(NC)"; \
		fi; \
		rm -f $(PID_DIR)/backend.pid; \
	fi
	@pkill -f "uvicorn backend.api.server:app" 2>/dev/null || true

stop-frontend:
	@if [ -f $(PID_DIR)/frontend.pid ]; then \
		PID=$$(cat $(PID_DIR)/frontend.pid); \
		if kill -0 $$PID 2>/dev/null; then \
			kill $$PID; \
			echo "$(GREEN)Frontend stopped (PID: $$PID)$(NC)"; \
		fi; \
		rm -f $(PID_DIR)/frontend.pid; \
	fi
	@pkill -f "next dev" 2>/dev/null || true

logs:
	@echo "$(CYAN)Showing backend logs (Ctrl+C to exit)...$(NC)"
	@tail -f logs/*.log 2>/dev/null || echo "$(YELLOW)No log files found$(NC)"

# ============================================================================
# Testing
# ============================================================================

test:
	@echo "$(CYAN)Running all tests (parallel)...$(NC)"
	$(PYTEST) -n auto -v

test-unit:
	@echo "$(CYAN)Running unit tests...$(NC)"
	$(PYTEST) tests/unit -v

test-integration:
	@echo "$(CYAN)Running integration tests...$(NC)"
	$(PYTEST) tests/integration -v

test-serial:
	@echo "$(CYAN)Running tests serially...$(NC)"
	$(PYTEST) -n 0 -v

test-coverage:
	@echo "$(CYAN)Running tests with coverage...$(NC)"
	$(PYTEST) --cov=backend --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(NC)"

test-watch:
	@echo "$(CYAN)Running tests in watch mode...$(NC)"
	$(PYTEST) -f -v

# ============================================================================
# Backtesting
# ============================================================================

# Easy backtest with real Yahoo Finance data
# Usage:
#   make backtest-live                    # SPY, 1 year
#   make backtest-live SYMBOL=AAPL        # Custom symbol
#   make backtest-live SYMBOLS=SPY,QQQ    # Compare multiple
#   make backtest-quick                   # Quick 6-month test

SYMBOL ?= SPY
SYMBOLS ?= SPY,QQQ,IWM,DIA
START_DATE ?=
END_DATE ?=

backtest-live:
	@echo "$(CYAN)Running backtest with live Yahoo Finance data...$(NC)"
ifdef SYMBOLS
	$(PYTHON) scripts/run_backtest.py --symbols $(SYMBOLS)
else ifdef START_DATE
	$(PYTHON) scripts/run_backtest.py --symbol $(SYMBOL) --start $(START_DATE)
else
	$(PYTHON) scripts/run_backtest.py --symbol $(SYMBOL)
endif

backtest-quick:
	@echo "$(CYAN)Running quick 6-month backtest...$(NC)"
	$(PYTHON) scripts/run_backtest.py --symbol $(SYMBOL) --quick

backtest-compare:
	@echo "$(CYAN)Comparing multiple symbols with Monte Carlo validation...$(NC)"
	$(PYTHON) scripts/run_backtest.py --symbols $(SYMBOLS) --monte-carlo --report

backtest-full:
	@echo "$(CYAN)Running full 2-year backtest analysis...$(NC)"
	$(PYTHON) scripts/run_backtest.py --symbol $(SYMBOL) --full

backtest-report:
	@echo "$(CYAN)Running backtest with HTML report + Monte Carlo validation...$(NC)"
	$(PYTHON) scripts/run_backtest.py --symbol $(SYMBOL) --report --monte-carlo --diagnostics

backtest-report-full:
	@echo "$(CYAN)Running full 2-year analysis with report + Monte Carlo...$(NC)"
	$(PYTHON) scripts/run_backtest.py --symbol $(SYMBOL) --full --report --monte-carlo --diagnostics

backtest-all:
	@echo "$(CYAN)Comparing ALL strategies on $(SYMBOL)...$(NC)"
	$(PYTHON) scripts/run_backtest.py --symbol $(SYMBOL) --all-strategies --report --monte-carlo

backtest-all-quick:
	@echo "$(CYAN)Quick comparison of ALL strategies on $(SYMBOL)...$(NC)"
	$(PYTHON) scripts/run_backtest.py --symbol $(SYMBOL) --all-strategies --quick

# Original synthetic data backtest
backtest:
	@echo "$(CYAN)Running SPY backtest with synthetic data...$(NC)"
	$(PYTHON) scripts/run_spy_backtest.py

backtest-optimize:
	@echo "$(CYAN)Running walk-forward grid search optimization...$(NC)"
	$(PYTHON) scripts/run_grid_search.py

# ============================================================================
# Notebooks
# ============================================================================

JUPYTER := $(VENV)/bin/jupyter

notebook: notebook-kernel
	@echo "$(CYAN)Starting Jupyter Notebook...$(NC)"
	@$(JUPYTER) notebook notebooks/

notebook-lab: notebook-kernel
	@echo "$(CYAN)Starting Jupyter Lab...$(NC)"
	@$(JUPYTER) lab notebooks/

notebook-kernel:
	@if [ ! -f $(JUPYTER) ]; then \
		echo "$(CYAN)Installing Jupyter (first time)...$(NC)"; \
		$(UV) sync; \
	fi
	@echo "$(CYAN)Registering Jupyter kernel...$(NC)"
	@$(PYTHON) -m ipykernel install --user --name=fluxhero --display-name="FluxHero"
	@echo "$(GREEN)Kernel 'FluxHero' registered$(NC)"

# ============================================================================
# Code Quality
# ============================================================================

lint:
	@echo "$(CYAN)Running ruff linter...$(NC)"
	$(RUFF) check backend/ tests/

format:
	@echo "$(CYAN)Formatting code with ruff...$(NC)"
	$(RUFF) format backend/ tests/
	$(RUFF) check --fix backend/ tests/
	@echo "$(GREEN)Code formatted$(NC)"

typecheck:
	@echo "$(CYAN)Running mypy type checker...$(NC)"
	$(MYPY) backend/

check: lint typecheck
	@echo "$(GREEN)All quality checks passed$(NC)"

# ============================================================================
# Installation
# ============================================================================

install: install-backend install-frontend
	@echo "$(GREEN)All dependencies installed$(NC)"

install-backend:
	@echo "$(CYAN)Installing Python dependencies with uv...$(NC)"
	$(UV) sync
	@echo "$(GREEN)Python dependencies installed$(NC)"

install-frontend:
	@echo "$(CYAN)Installing Node.js dependencies...$(NC)"
	cd frontend && npm install
	@echo "$(GREEN)Node.js dependencies installed$(NC)"

# ============================================================================
# Maintenance
# ============================================================================

daily-reboot:
	@echo "$(CYAN)Running daily maintenance...$(NC)"
	$(PYTHON) -m backend.maintenance.daily_reboot

archive-trades:
	@echo "$(CYAN)Archiving old trades...$(NC)"
	$(PYTHON) -c "import asyncio; from backend.storage.sqlite_store import SQLiteStore; \
		async def archive(): \
			store = SQLiteStore(); \
			await store.initialize(); \
			count = await store.archive_old_trades(30); \
			print(f'Archived {count} trades'); \
			await store.close(); \
		asyncio.run(archive())"

seed-data:
	@echo "$(CYAN)Seeding test data...$(NC)"
	$(PYTHON) scripts/seed_test_data.py

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "$(CYAN)Cleaning up...$(NC)"
	rm -rf $(PID_DIR)
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete$(NC)"

clean-all: clean
	@echo "$(CYAN)Deep cleaning (includes node_modules and venv)...$(NC)"
	rm -rf $(VENV)
	rm -rf frontend/node_modules frontend/.next
	@echo "$(GREEN)Deep cleanup complete$(NC)"

# ============================================================================
# Docker
# ============================================================================

docker-build:
	@echo "$(CYAN)Building Docker images...$(NC)"
	docker compose build
	@echo "$(GREEN)Docker images built successfully$(NC)"

docker-up:
	@echo "$(CYAN)Starting Docker containers...$(NC)"
	docker compose up -d
	@echo "$(GREEN)Containers started$(NC)"
	@echo "  Backend:  http://localhost:$(BACKEND_PORT)"
	@echo "  Frontend: http://localhost:$(FRONTEND_PORT)"
	@echo "  API Docs: http://localhost:$(BACKEND_PORT)/docs"

docker-down:
	@echo "$(CYAN)Stopping Docker containers...$(NC)"
	docker compose down
	@echo "$(GREEN)Containers stopped$(NC)"

docker-logs:
	@echo "$(CYAN)Following Docker logs (Ctrl+C to exit)...$(NC)"
	docker compose logs -f

docker-shell-backend:
	@echo "$(CYAN)Opening shell in backend container...$(NC)"
	docker compose exec backend bash

docker-clean:
	@echo "$(CYAN)Removing containers, volumes, and images...$(NC)"
	docker compose down -v --rmi all
	@echo "$(GREEN)Docker cleanup complete$(NC)"
