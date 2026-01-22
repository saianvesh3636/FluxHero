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

.PHONY: help dev dev-backend dev-frontend stop stop-backend stop-frontend \
        test test-unit test-integration test-parallel lint format typecheck \
        install install-backend install-frontend clean logs

# Colors for terminal output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Configuration
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
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
	@echo "  make dev-backend      Start backend only (port $(BACKEND_PORT))"
	@echo "  make dev-frontend     Start frontend only (port $(FRONTEND_PORT))"
	@echo "  make stop             Stop all running services"
	@echo "  make logs             Show backend logs"
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
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install          Install all dependencies"
	@echo "  make install-backend  Install Python dependencies"
	@echo "  make install-frontend Install Node.js dependencies"
	@echo "  make clean            Remove generated files and caches"
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
	@echo "$(CYAN)Installing Python dependencies...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
		echo "$(GREEN)Virtual environment created$(NC)"; \
	fi
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
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
