# FluxHero Enhancement Tasks - SentinelTrader Comparison

Tasks derived from comparing FluxHero with SentinelTrader. Each task is on a single line for ralphy.sh compatibility.

See `docs/COMPARISON_SENTINEL_TRADER.md` for full comparison analysis.

**IMPORTANT**: SentinelTrader uses Flask + React (Vite) + SQLAlchemy. Our stack is FastAPI + Next.js + raw SQLite. Do NOT copy code from SentinelTrader - use it as conceptual reference only and implement using our patterns.

---

## Phase A: Multi-Broker Architecture (HIGH PRIORITY)

- [x] Create broker abstraction base class (backend/execution/broker_base.py) - define abstract BrokerInterface with methods: connect(), disconnect(), get_account(), get_positions(), place_order(), cancel_order(), get_order_status(), health_check(), use Python ABC for interface definition. Use async methods compatible with FastAPI. Concept ref: SentinelTrader broker_factory.py (Flask/SQLAlchemy - adapt to our FastAPI/Pydantic patterns).

- [x] Implement Alpaca broker adapter (backend/execution/brokers/alpaca_broker.py) - refactor existing broker_interface.py into adapter pattern implementing BrokerInterface, handle Alpaca-specific auth (API key + secret), map Alpaca responses to unified Pydantic models, add connection pooling and retry logic using httpx. Ref: existing backend/execution/broker_interface.py

- [x] Create broker factory (backend/execution/broker_factory.py) - implement factory pattern with create_broker(broker_type: str, config: dict) -> BrokerInterface, support types: "alpaca", validate config per broker type using Pydantic, singleton pattern for connection reuse. Don't create Paper and mock yet. Concept ref: SentinelTrader broker_factory.py (Flask - adapt to FastAPI async patterns).

- [x] Add broker credential encryption (backend/execution/broker_credentials.py) - implement AES encryption for API keys/secrets using cryptography library, use FLUXHERO_ENCRYPTION_KEY from env via backend/core/config.py, encrypt_credential() and decrypt_credential() functions, never log decrypted values. Concept ref: SentinelTrader broker.py encrypt_api_key() (SQLAlchemy model - adapt to our dataclass/Pydantic approach).

- [x] Add broker API endpoints (backend/api/server.py) - add GET /api/brokers (list configured brokers), POST /api/brokers (add broker config), DELETE /api/brokers/{id}, GET /api/brokers/{id}/health (connection health check), use Pydantic models for request/response validation following existing server.py patterns. Concept ref: SentinelTrader broker.py routes (Flask - implement as FastAPI async endpoints).

- [x] Add broker selection to frontend (frontend/app/settings/page.tsx) - create settings page with broker configuration form, dropdown for broker type selection, secure credential input fields (password type), connection test button, display connection status. Use Next.js App Router and Tailwind CSS following existing frontend patterns. Concept ref: SentinelTrader BrokerConnectionManager.tsx (React/Redux - adapt to our Next.js/hooks approach).

- [x] Create broker integration tests (tests/integration/test_broker_adapters.py) - test each broker adapter with mock responses, test factory creates correct broker type, test credential encryption/decryption round-trip, test connection retry logic, test error handling for network failures. Use pytest-asyncio for async tests.

---

## Phase B: Paper Trading System (HIGH PRIORITY)

- [x] Implement paper broker adapter (backend/execution/brokers/paper_broker.py) - create PaperBroker implementing BrokerInterface from Phase A, auto-create $100,000 paper account, store account state in SQLite via backend/storage/sqlite_store.py (balance, positions, trades), reset_account() method to restore initial state, track realized and unrealized P&L. Concept ref: SentinelTrader paper_broker.py (Flask - implement as async class matching our broker interface).

- [x] Add slippage simulation (backend/execution/brokers/paper_broker.py) - configurable slippage in basis points (default 5 bps via FLUXHERO_PAPER_SLIPPAGE_BPS env), apply slippage on order fill: buy_price = price * (1 + slippage_bps/10000), sell_price = price * (1 - slippage_bps/10000), log slippage impact using loguru. Concept ref: SentinelTrader _apply_slippage() (adapt formula to our implementation).

- [ ] Add market price simulation (backend/execution/brokers/paper_broker.py) - fetch last known price from backend/data/yahoo_provider.py for realistic fills, cache prices with 1-minute TTL using simple dict cache, fallback to configurable mock prices if fetch fails, support price override for testing via method parameter.

- [ ] Add paper trading API endpoints (backend/api/server.py) - add GET /api/paper/account (balance, positions), POST /api/paper/reset (reset to initial state), GET /api/paper/trades (paper trade history), return same Pydantic response models as live broker for UI compatibility. Follow existing FastAPI endpoint patterns in server.py.

- [ ] Add paper/live toggle to frontend (frontend/components/TradingModeToggle.tsx) - create toggle component for paper vs live mode, prominent visual indicator (green for paper, red for live) using Tailwind classes, confirmation dialog when switching to live, persist selection in localStorage. Use React useState/useEffect hooks. Concept ref: SentinelTrader SystemBanner.tsx (React/Redux - adapt to our hooks-based approach).

- [ ] Create paper trading tests (tests/integration/test_paper_trading.py) - test account initialization with correct balance, test order placement updates positions, test slippage applied correctly, test P&L calculation accuracy, test account reset functionality. Use pytest with pytest-asyncio.

---

## Phase C: Docker Deployment (HIGH PRIORITY)

- [ ] Create backend Dockerfile (docker/Dockerfile.backend) - use python:3.11-slim base image, install uv package manager (curl -LsSf https://astral.sh/uv/install.sh), copy pyproject.toml and uv.lock, run uv sync, copy backend code, expose port 8000, CMD uvicorn backend.api.server:app, add HEALTHCHECK for /api/health endpoint.

- [ ] Create frontend Dockerfile (docker/Dockerfile.frontend) - use node:20-alpine base image, copy package.json and package-lock.json, run npm ci, copy frontend code, run npm run build (next build), expose port 3000, CMD npm start (next start). Multi-stage build to reduce image size.

- [ ] Create docker-compose.yml (docker-compose.yml) - define services: backend (port 8000), frontend (port 3000), volumes for data persistence (./data:/app/data for sqlite db and parquet cache, ./logs:/app/logs), env_file: .env, health checks using curl, depends_on with condition: service_healthy for startup order.

- [ ] Create .dockerignore files (.dockerignore) - exclude node_modules, __pycache__, .venv, .git, .env (use .env.example), tests/, docs/, *.md, .pytest_cache, .mypy_cache, IDE configs (.vscode, .idea). Create separate backend and frontend .dockerignore if needed.

- [ ] Add Docker commands to Makefile (Makefile) - add docker-build (docker compose build), docker-up (docker compose up -d), docker-down (docker compose down), docker-logs (docker compose logs -f), docker-shell-backend (docker compose exec backend bash), docker-clean (docker compose down -v --rmi all). Follow existing Makefile patterns.

- [ ] Create docker environment template (docker/.env.docker.example) - document all required FLUXHERO_* env vars, set FLUXHERO_DB_PATH=/app/data/fluxhero.db, FLUXHERO_CACHE_DIR=/app/data/cache, FLUXHERO_LOG_FILE=/app/logs/fluxhero.log, API URLs for inter-service communication (http://backend:8000).

- [ ] Add Docker deployment docs (docs/DOCKER_DEPLOYMENT.md) - document build and run process with docker compose, explain volume mounts for data persistence, nginx reverse proxy setup for SSL/TLS, production vs development configs (FLUXHERO_ENV=production), troubleshooting common issues (permissions, networking).

---

## Phase D: Audit Logging (MEDIUM PRIORITY)

- [ ] Create audit log model (backend/storage/audit_log.py) - create AuditLog dataclass with id (uuid), timestamp (datetime), user_id (optional str), action (str), resource (str), resource_id (optional str), old_value (optional json str), new_value (optional json str), ip_address (optional str), add to sqlite_store.py with CREATE TABLE and index on timestamp. Concept ref: SentinelTrader audit_log.py (SQLAlchemy - adapt to our raw SQLite/dataclass approach).

- [ ] Implement audit logger (backend/core/audit_logger.py) - create async log_audit(action, resource, resource_id, old_value, new_value) function, use FastAPI Request context for user/IP extraction, serialize values to JSON, async write to SQLite to avoid blocking, integrate with existing loguru logging. Concept ref: SentinelTrader audit_service.py (Flask - adapt to FastAPI async patterns).

- [ ] Add audit logging to critical operations (various files) - log all trade entries/exits in backend/execution/order_manager.py, log broker config changes in broker API endpoints, log system events in backend/maintenance/daily_reboot.py. Use audit_logger.log_audit() calls.

- [ ] Add audit log API endpoint (backend/api/server.py) - add GET /api/audit/logs with Pydantic query params: limit (int, default 100), offset (int, default 0), start_date (optional datetime), end_date (optional datetime), action (optional str), return AuditLogResponse with items list and total count.

- [ ] Add audit log viewer to frontend (frontend/app/audit/page.tsx) - create table displaying audit logs with columns (timestamp, action, resource, user), date range filter using react-datepicker, action type dropdown filter, pagination controls (prev/next buttons), export to CSV button using Blob download. Use Next.js App Router and Tailwind. Concept ref: SentinelTrader AuditLogs.tsx (React/Redux - adapt to our Next.js/hooks approach).

---

## Phase E: Database Migrations (MEDIUM PRIORITY)

- [ ] Add Alembic for migrations (backend/storage/migrations/) - add alembic to pyproject.toml dependencies, run alembic init backend/storage/migrations, configure alembic.ini with sqlalchemy.url = sqlite:///data/fluxhero.db, update env.py to import our models, set target_metadata for autogenerate.

- [ ] Create initial migration (backend/storage/migrations/versions/001_initial.py) - run alembic revision --autogenerate -m "initial" to capture existing tables (trades, positions, candles, settings from sqlite_store.py), review generated migration, ensure upgrade() and downgrade() are correct, test migration on fresh db.

- [ ] Add migration commands to Makefile (Makefile) - add db-migrate (alembic upgrade head), db-revision (alembic revision --autogenerate -m), db-downgrade (alembic downgrade -1), db-history (alembic history), db-current (alembic current). Follow existing Makefile target patterns.

- [ ] Document migration workflow (docs/DATABASE_MIGRATIONS.md) - explain how to create new migrations after model changes, how to run migrations in dev (make db-migrate) vs prod (docker compose exec backend alembic upgrade head), how to handle migration conflicts, backup recommendations (cp data/fluxhero.db data/fluxhero.db.bak).

---

## Phase F: Manual Trading Widget (MEDIUM PRIORITY)

- [ ] Create manual trade API endpoints (backend/api/server.py) - add POST /api/trade/manual with Pydantic model: symbol (str), side (Literal["buy", "sell"]), quantity (int), order_type (Literal["market", "limit"]), limit_price (optional float), validate against risk limits using backend/risk/position_limits.py, call broker.place_order(), return TradeResponse. Concept ref: SentinelTrader trading.py routes (Flask - implement as FastAPI async endpoint).

- [ ] Add position sizing calculator endpoint (backend/api/server.py) - add POST /api/trade/calculate-size with Pydantic model: symbol (str), risk_percent (float), stop_loss_price (float), use backend/execution/position_sizer.py calculate_position_size(), return PositionSizeResponse with shares (int), estimated_cost (float), risk_amount (float).

- [ ] Create manual trading widget (frontend/components/ManualTradeWidget.tsx) - create form with symbol input (text with validation), side toggle buttons (Buy green/Sell red), quantity number input, order type radio (Market/Limit), limit price input (shown only for limit orders), cost estimate display, submit button with loading state. Use React useState, Tailwind styling. Concept ref: SentinelTrader EnhancedManualTradeWidget.tsx (React/Redux - adapt to our hooks-based approach).

- [ ] Add quick size buttons (frontend/components/ManualTradeWidget.tsx) - add preset buttons for 25%, 50%, 75%, 100% of max position from /api/trade/calculate-size response, style as button group with Tailwind, update quantity input onChange, disable 100% if exceeds risk limits, show tooltip explaining calculation.

- [ ] Add risk visualization (frontend/components/RiskIndicator.tsx) - create component showing risk level based on quantity: Low (<0.5% account, green), Medium (0.5-1% account, yellow), High (>1% account, red), use Tailwind bg colors, show progress bar with risk percentage, update in real-time as quantity changes via props. Concept ref: SentinelTrader RiskIndicator.tsx (React - adapt styling to Tailwind).

- [ ] Add manual trading to live page (frontend/app/live/page.tsx) - integrate ManualTradeWidget component into existing live trading page, position in right sidebar or below positions table, fetch broker status to enable/disable widget, show confirmation modal (react portal) before order submission with order details.

---

## Dependencies

- Phase B (Paper Trading) depends on Phase A (Multi-Broker) for BrokerInterface abstraction
- Phase D (Audit Logging) is standalone but benefits from having broker/trade operations to log
- Phase F (Manual Trading) can use audit logging from Phase D for trade logging

---

## Tech Stack Reference

| Component | FluxHero (This Project) | SentinelTrader (Reference Only) |
|-----------|------------------------|--------------------------------|
| Backend Framework | FastAPI (async) | Flask (sync) |
| API Validation | Pydantic models | Marshmallow schemas |
| Database | Raw SQLite + dataclasses | SQLAlchemy ORM |
| Frontend Framework | Next.js App Router | React + Vite |
| State Management | React hooks (useState, useEffect) | Redux Toolkit |
| Styling | Tailwind CSS | Tailwind CSS |
| Package Manager | uv | uv (recently migrated) |
| Testing | pytest + pytest-asyncio | pytest |

**Remember**: Reference SentinelTrader for concepts and features, but implement using FluxHero patterns found in existing codebase.
