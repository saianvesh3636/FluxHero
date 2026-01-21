# FluxHero System Operations Guide

**Version**: 1.0
**Last Updated**: 2026-01-21

This guide covers daily operations, maintenance tasks, monitoring, and troubleshooting for the FluxHero adaptive retail quant trading system.

---

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [System Startup](#system-startup)
3. [System Shutdown](#system-shutdown)
4. [Monitoring](#monitoring)
5. [Maintenance Tasks](#maintenance-tasks)
6. [Troubleshooting](#troubleshooting)
7. [Emergency Procedures](#emergency-procedures)
8. [Backup and Recovery](#backup-and-recovery)

---

## Daily Operations

### 9:00 AM EST Daily Reboot (Automated)

The system automatically performs a daily reboot at 9:00 AM EST to ensure fresh state before market open (9:30 AM EST).

**What happens during reboot:**
1. System closes all WebSocket connections
2. Flushes in-memory candle buffer
3. Fetches latest 500 candles from API (or cache if <24h old)
4. Reconnects WebSocket for live price feeds
5. Reinitializes indicators and regime detection
6. Resumes normal operation

**Automated script location**: `fluxhero/scripts/daily_reboot.py`

**Manual execution** (if needed):
```bash
cd /Users/anvesh/Developer/QuantTrading/project/fluxhero
python scripts/daily_reboot.py
```

**Pre-market checklist** (before 9:30 AM):
- [ ] Verify daily reboot completed successfully (check logs)
- [ ] Confirm WebSocket connection is active (green status indicator)
- [ ] Validate 500 candles loaded in buffer
- [ ] Check system status endpoint: `curl http://localhost:8000/api/status`
- [ ] Review overnight alerts or errors in logs

---

## System Startup

### Backend Startup

**Production mode:**
```bash
cd /Users/anvesh/Developer/QuantTrading/project/fluxhero
source venv/bin/activate
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000
```

**Development mode** (with auto-reload):
```bash
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000 --reload
```

**Startup sequence** (automatic):
1. Load system settings from SQLite database
2. Initialize SQLite store (trades, positions, settings)
3. Load cached candles from Parquet (if fresh <24h)
4. If cache stale or missing: Fetch 500 candles from API
5. Initialize 500-candle rolling buffer
6. Calculate initial indicators (EMA, ATR, RSI, KAMA)
7. Detect initial market regime (trend/mean-reversion/neutral)
8. Connect WebSocket for live price feeds
9. Start FastAPI server on port 8000
10. Begin processing live ticks

**Expected startup time**: <3 seconds (with cache hit)

**Verify successful startup:**
```bash
# Check system status
curl http://localhost:8000/api/status

# Expected response:
{
  "status": "active",
  "websocket_connected": true,
  "last_tick_time": "2026-01-21T14:30:15Z",
  "buffer_size": 500,
  "uptime_seconds": 45
}
```

### Frontend Startup

**Development mode:**
```bash
cd /Users/anvesh/Developer/QuantTrading/project/fluxhero/frontend
npm run dev
```

**Production build:**
```bash
npm run build
npm run start
```

**Access dashboard**: http://localhost:3000

---

## System Shutdown

### Graceful Shutdown

**Backend:**
```bash
# Press Ctrl+C in the terminal running uvicorn
# Wait for graceful shutdown message
```

**What happens during shutdown:**
1. Stop accepting new WebSocket connections
2. Close existing WebSocket connections
3. Flush pending async writes to SQLite
4. Close database connections
5. Archive trade data if needed
6. Write final state to disk

**Force shutdown** (emergency only):
```bash
pkill -f "uvicorn backend.api.server"
```

### Pre-Shutdown Checklist
- [ ] Close all open positions (or verify stops are in place)
- [ ] Cancel all pending orders
- [ ] Export recent trade history to CSV (if needed)
- [ ] Verify no critical alerts pending
- [ ] Note reason for shutdown in logs

---

## Monitoring

### Real-Time Monitoring

**System health indicators:**
1. **WebSocket status**: 🟢 Active / 🟡 Delayed / 🔴 Offline
2. **Heartbeat monitor**: Alert if no data for >60 seconds
3. **Daily P&L**: Current profit/loss for the trading day
4. **Current drawdown**: Percentage decline from peak equity
5. **Total exposure**: Percentage of capital deployed

**Dashboard tabs:**
- **Live Trading**: Open positions, real-time P&L, system status
- **Analytics**: Charts, indicators, regime state
- **Trade History**: Recent trades with filters
- **Backtesting**: Run backtests and view tearsheets

### Log Files

**Location**: `fluxhero/logs/`

**Important log files:**
- `fluxhero.log`: Main application log
- `trades.log`: Trade execution log
- `errors.log`: Error and exception log
- `api.log`: API request/response log

**Tail logs in real-time:**
```bash
tail -f fluxhero/logs/fluxhero.log
```

**Search for errors:**
```bash
grep ERROR fluxhero/logs/fluxhero.log | tail -20
```

### Key Metrics to Monitor

**Performance metrics:**
- Win rate (target: >45%)
- Sharpe ratio (target: >0.8)
- Max drawdown (target: <25%)
- Daily P&L volatility

**Operational metrics:**
- WebSocket uptime (target: >99%)
- API error rate (target: <1%)
- Signal generation latency (target: <100ms)
- Order fill latency (target: <500ms)

**Risk metrics:**
- Current drawdown vs max allowed (20%)
- Total exposure vs limit (50%)
- Position concentration (max 20% per position)
- Correlation between open positions

---

## Maintenance Tasks

### Daily Maintenance (Automated)

**9:00 AM EST - Daily reboot** (automated via cron/scheduler)
- System restart as described above
- No manual intervention required

**4:30 PM EST - End-of-day archival** (automated)
- Archive closed trades older than 30 days
- Flush trade data to Parquet for long-term storage
- Update daily performance metrics
- Generate daily performance report

### Weekly Maintenance

**Every Monday, 6:00 AM EST:**
- [ ] Review performance metrics for previous week
- [ ] Check for any degraded indicators or alerts
- [ ] Verify database size is within limits (<100 MB for SQLite)
- [ ] Clear old log files (keep last 30 days)
- [ ] Review regime detection accuracy
- [ ] Validate cache freshness and cleanup stale caches

**Log rotation:**
```bash
cd /Users/anvesh/Developer/QuantTrading/project/fluxhero/logs
find . -name "*.log" -mtime +30 -delete
```

**Database cleanup:**
```bash
cd /Users/anvesh/Developer/QuantTrading/project/fluxhero
python -c "from backend.storage.sqlite_store import SQLiteStore; store = SQLiteStore(); store.archive_old_trades(days=30); store.close()"
```

### Monthly Maintenance

**First Sunday of each month:**
- [ ] Full system backup (database + config + logs)
- [ ] Dependency security audit: `pip list --outdated`
- [ ] Review and update risk parameters if needed
- [ ] Performance attribution analysis (which strategies contributed to P&L)
- [ ] Regime detection calibration check
- [ ] Walk-forward test validation (update training windows)

**Database vacuum** (compact and optimize):
```bash
sqlite3 fluxhero/data/system.db "VACUUM;"
```

**Parquet cache cleanup:**
```bash
# Delete caches older than 7 days
find fluxhero/data/cache -name "*.parquet" -mtime +7 -delete
```

### Quarterly Maintenance

**Every 3 months:**
- [ ] Full strategy backtest (1-year historical data)
- [ ] Validate metrics meet targets (Sharpe >0.8, DD <25%, Win Rate >45%)
- [ ] Review and adjust position sizing rules if needed
- [ ] Update dependency versions (Python packages, npm packages)
- [ ] Security review (API keys rotation, access logs)
- [ ] Disaster recovery drill (restore from backup)

---

## Troubleshooting

### Common Issues

#### 1. WebSocket Connection Lost

**Symptoms:**
- 🔴 Offline status indicator
- No live price updates
- Stale connection warning in logs

**Solution:**
```bash
# Check WebSocket status
curl http://localhost:8000/api/status

# Automatic reconnect should trigger within 5 seconds
# If not, restart backend:
# Ctrl+C to stop, then restart uvicorn
```

**Root causes:**
- Network connectivity issues
- Broker API downtime
- API rate limit exceeded
- Authentication token expired

#### 2. High Slippage on Orders

**Symptoms:**
- Fill prices significantly worse than expected
- Trade P&L underperforming backtest results

**Solution:**
- Check volume validation in noise filter (may be trading illiquid stocks)
- Review SV ratio (spread-to-volatility) for recent trades
- Consider widening order limits or using different order types
- Avoid trading during illiquid hours (pre-market, lunch, after-hours)

#### 3. System Startup Fails

**Symptoms:**
- Error during startup sequence
- uvicorn crashes or hangs

**Common fixes:**
```bash
# Check if port 8000 is already in use
lsof -i :8000
# Kill process if needed: kill -9 <PID>

# Verify database integrity
sqlite3 fluxhero/data/system.db "PRAGMA integrity_check;"

# Delete stale cache and retry
rm -rf fluxhero/data/cache/*

# Check API credentials
cat fluxhero/.env  # Verify ALPACA_API_KEY and ALPACA_SECRET_KEY
```

#### 4. Indicator Calculation Performance Degradation

**Symptoms:**
- Indicator calculations taking >500ms for 10k candles
- System feels sluggish

**Solution:**
- Check if Numba JIT cache is corrupted: `rm -rf ~/.numba_cache`
- Verify no infinite loops in custom indicators
- Monitor memory usage: `top -pid <uvicorn_pid>`
- Restart system to rebuild JIT cache

#### 5. Database Lock Errors

**Symptoms:**
- SQLite errors: "database is locked"
- Slow trade logging

**Solution:**
- Verify only one instance of backend is running
- Check for hung async write tasks
- Restart backend to clear locks
- If persistent, vacuum database: `sqlite3 fluxhero/data/system.db "VACUUM;"`

#### 6. Regime Detection Whipsaws

**Symptoms:**
- Frequent regime switches (TREND ↔ MEAN_REVERSION)
- Poor strategy performance due to mode switching

**Solution:**
- Increase regime persistence window (default: 3 bars)
- Adjust ADX thresholds (trending >25, ranging <20)
- Adjust R² thresholds (trend >0.6, no trend <0.4)
- Review recent market conditions (may be genuinely transitioning)

---

## Emergency Procedures

### Kill Switch Activation

**Automatic triggers** (kill switch activates automatically):
1. Daily loss exceeds 3% of capital
2. Drawdown reaches 20% from peak equity
3. Position concentration exceeds 50% in single position
4. Critical system error (database corruption, API failure)

**What happens during kill switch:**
1. All new order submissions blocked
2. All open positions closed at market price
3. Pending orders cancelled
4. System enters safe mode (monitoring only)
5. Alert sent to operator

**Manual kill switch activation:**
```bash
# Emergency position closure
curl -X POST http://localhost:8000/api/emergency/close_all_positions

# Or via Python:
cd /Users/anvesh/Developer/QuantTrading/project/fluxhero
python -c "from backend.risk.kill_switch import trigger_kill_switch; trigger_kill_switch()"
```

### Recovery from Kill Switch

**Steps to resume trading:**
1. Identify and resolve root cause (excessive losses, system error, etc.)
2. Review recent trades and logs
3. Adjust risk parameters if needed
4. Reset kill switch: `curl -X POST http://localhost:8000/api/emergency/reset_kill_switch`
5. Verify system status is green before resuming
6. Monitor closely for first hour after reset

---

## Backup and Recovery

### Backup Strategy

**What to backup:**
1. SQLite database: `fluxhero/data/system.db`
2. Configuration files: `fluxhero/.env`, `fluxhero/config/`
3. Custom indicators/strategies (if modified)
4. Recent log files (last 7 days)

**Backup frequency:**
- **Daily**: Automated incremental backup of SQLite database
- **Weekly**: Full backup of all data and config
- **Monthly**: Off-site backup to cloud storage

**Daily automated backup** (cron job):
```bash
# Add to crontab: crontab -e
0 17 * * * /Users/anvesh/Developer/QuantTrading/project/fluxhero/scripts/backup_daily.sh
```

**Manual backup:**
```bash
cd /Users/anvesh/Developer/QuantTrading/project/fluxhero
mkdir -p backups/$(date +%Y%m%d)
cp data/system.db backups/$(date +%Y%m%d)/system.db
cp -r config backups/$(date +%Y%m%d)/
cp .env backups/$(date +%Y%m%d)/
tar -czf backups/fluxhero_backup_$(date +%Y%m%d).tar.gz backups/$(date +%Y%m%d)
```

### Recovery Procedures

**Restore from backup:**
```bash
# Stop system
# Ctrl+C in uvicorn terminal

# Restore database
cd /Users/anvesh/Developer/QuantTrading/project/fluxhero
cp backups/20260121/system.db data/system.db

# Restore config
cp backups/20260121/.env .env

# Restart system
source venv/bin/activate
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000
```

**Disaster recovery** (full system rebuild):
1. Clone repository: `git clone <repo_url>`
2. Restore latest database backup
3. Restore configuration files
4. Install dependencies: `pip install -r requirements.txt`
5. Verify API credentials
6. Run integration test: `pytest tests/integration/`
7. Start system and verify status

---

## Appendix

### Useful Commands

**Check system resource usage:**
```bash
# CPU and memory
top -pid $(pgrep -f uvicorn)

# Disk usage
du -sh fluxhero/data/*

# Database size
ls -lh fluxhero/data/system.db
```

**Database queries:**
```bash
# Count open positions
sqlite3 fluxhero/data/system.db "SELECT COUNT(*) FROM positions;"

# Recent trades
sqlite3 fluxhero/data/system.db "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;"

# Daily P&L
sqlite3 fluxhero/data/system.db "SELECT DATE(entry_time), SUM(realized_pnl) FROM trades WHERE status=1 GROUP BY DATE(entry_time) ORDER BY entry_time DESC LIMIT 7;"
```

**Performance profiling:**
```bash
# Profile indicator calculations
python -m cProfile -s cumtime backend/computation/indicators.py

# Measure WebSocket latency
ping <broker_websocket_host>
```

### Contact Information

**System Administrator**: anvesh@example.com
**Emergency Contact**: +1-XXX-XXX-XXXX
**Broker Support**: support@alpaca.markets

### Maintenance Schedule Summary

| Frequency | Time (EST) | Task | Duration |
|-----------|------------|------|----------|
| Daily | 9:00 AM | Automated reboot | 5 min |
| Daily | 4:30 PM | End-of-day archival | 2 min |
| Weekly | Monday 6:00 AM | Weekly maintenance | 30 min |
| Monthly | 1st Sunday | Full backup + audit | 2 hours |
| Quarterly | - | Strategy review + backtest | 4 hours |

---

**Document History:**
- v1.0 (2026-01-21): Initial creation for Phase 16 completion
