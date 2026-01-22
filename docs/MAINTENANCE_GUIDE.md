# FluxHero Maintenance Guide

**Version**: 1.0.0
**Last Updated**: 2026-01-21
**Target**: Production System Maintenance

---

## Table of Contents

1. [Overview](#overview)
2. [Dependency Management](#dependency-management)
3. [Data Backup & Recovery](#data-backup--recovery)
4. [System Monitoring](#system-monitoring)
5. [Database Maintenance](#database-maintenance)
6. [Log Management](#log-management)
7. [Performance Optimization](#performance-optimization)
8. [Security Updates](#security-updates)
9. [Disaster Recovery](#disaster-recovery)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance Schedules](#maintenance-schedules)

---

## Overview

This guide covers ongoing maintenance tasks for FluxHero in production. Regular maintenance ensures system reliability, performance, and data integrity.

### Maintenance Principles

- **Automate everything possible** - Use cron jobs and scripts
- **Test in development first** - Never apply updates directly to production
- **Backup before changes** - Always backup data before updates
- **Monitor continuously** - Track system health 24/7
- **Document changes** - Keep a maintenance log

### Key Maintenance Areas

| Area | Frequency | Criticality |
|------|-----------|-------------|
| Dependency updates | Monthly | High |
| Data backups | Daily | Critical |
| Database maintenance | Weekly | Medium |
| Log rotation | Daily | Low |
| Security patches | As needed | Critical |
| Performance tuning | Quarterly | Medium |

---

## Dependency Management

### Python Dependencies

#### Checking for Updates

```bash
# Activate virtual environment
cd /Users/anvesh/Developer/QuantTrading/project
source venv/bin/activate

# Check outdated packages
pip list --outdated

# Check security vulnerabilities
pip install pip-audit
pip-audit
```

#### Update Strategy

**Monthly Update Cycle** (First week of each month):

1. **Create test environment**:
```bash
# Clone production environment
python -m venv venv_test
source venv_test/bin/activate
pip install -r requirements.txt
```

2. **Update non-critical packages**:
```bash
# Update utilities and development tools (lowest risk)
pip install --upgrade pytest pytest-asyncio pytest-cov ruff mypy
pip install --upgrade python-dotenv pydantic pydantic-settings

# Run tests
pytest tests/ -v
```

3. **Update data processing packages** (medium risk):
```bash
# Update quantitative libraries
pip install --upgrade quantstats scipy scikit-learn

# Test backtesting module
pytest tests/backtesting/ -v
```

4. **Update core dependencies** (highest risk - test thoroughly):
```bash
# Update performance-critical packages
pip install --upgrade numba numpy pandas

# Run full test suite
pytest tests/ -v
pytest tests/computation/ --benchmark
```

5. **Update web framework** (high risk):
```bash
# Update API framework
pip install --upgrade fastapi uvicorn httpx websockets

# Test API endpoints
pytest tests/api/ -v
```

6. **Freeze new versions**:
```bash
pip freeze > requirements_new.txt

# Compare versions
diff requirements.txt requirements_new.txt
```

7. **Deploy to staging**:
```bash
# On staging server
scp requirements_new.txt user@staging:/path/to/fluxhero/
ssh user@staging
cd /path/to/fluxhero
pip install -r requirements_new.txt
systemctl restart fluxhero-backend
```

8. **Monitor staging for 48 hours**:
- Check error logs
- Verify trade execution
- Monitor performance metrics
- Validate backtest results

9. **Deploy to production** (if staging is stable):
```bash
# Backup first
./scripts/backup_all.sh

# Update production
scp requirements_new.txt user@production:/path/to/fluxhero/
ssh user@production
cd /path/to/fluxhero
source venv/bin/activate
pip install -r requirements_new.txt

# Restart services
sudo systemctl restart fluxhero-backend
sudo systemctl restart fluxhero-frontend
```

#### Critical Package Monitoring

**Numba** (performance-critical):
- Monitor: https://github.com/numba/numba/releases
- Breaking changes affect JIT compilation
- Test indicator calculations after updates

**FastAPI** (API stability):
- Monitor: https://github.com/tiangolo/fastapi/releases
- Test WebSocket connections after updates
- Verify CORS and authentication

**Pandas/Numpy** (data integrity):
- Test data pipeline thoroughly
- Verify backtest results match previous runs
- Check for deprecated function warnings

#### Rollback Procedure

If issues occur after update:

```bash
# Restore previous requirements
cp requirements.txt.backup requirements.txt
pip install -r requirements.txt

# Restart services
sudo systemctl restart fluxhero-backend
sudo systemctl restart fluxhero-frontend

# Verify system functionality
curl http://localhost:8000/api/status
```

### Frontend Dependencies (Node.js)

#### Checking for Updates

```bash
cd fluxhero/frontend

# Check outdated packages
npm outdated

# Check security vulnerabilities
npm audit

# Check for high/critical vulnerabilities only
npm audit --audit-level=high
```

#### Update Strategy

**Monthly Update Cycle**:

1. **Update development dependencies** (lowest risk):
```bash
# Update testing libraries
npm update @testing-library/jest-dom @testing-library/react
npm update @types/jest @types/node @types/react

# Update TypeScript
npm update typescript

# Run tests
npm test
```

2. **Update build tools**:
```bash
# Update Next.js (test thoroughly)
npm update next

# Build and test
npm run build
npm run start
```

3. **Update React** (high risk - major version changes):
```bash
# Check current version
npm list react react-dom

# Update to latest compatible version
npm update react react-dom

# Test all components
npm test
npm run build
```

4. **Update charting library**:
```bash
# Update lightweight-charts
npm update lightweight-charts

# Test chart rendering
npm run dev
# Manually test analytics page
```

5. **Security updates** (apply immediately):
```bash
# Fix vulnerabilities automatically
npm audit fix

# For breaking changes, update manually
npm audit fix --force  # Use with caution
```

#### Lockfile Management

```bash
# After updates, commit new lockfile
git add package.json package-lock.json
git commit -m "chore: Update frontend dependencies (2026-02)"

# On server, use exact versions from lockfile
npm ci  # Instead of npm install
```

### Dependency Update Checklist

Before deploying updates:

- [ ] Backup database and configuration
- [ ] Update in test environment first
- [ ] Run full test suite (backend and frontend)
- [ ] Run backtests and compare metrics with previous results
- [ ] Test API endpoints manually
- [ ] Test WebSocket connections
- [ ] Check logs for warnings/deprecations
- [ ] Monitor performance benchmarks
- [ ] Test in staging environment for 48 hours
- [ ] Document changes in maintenance log
- [ ] Have rollback plan ready

---

## Data Backup & Recovery

### Backup Strategy

FluxHero uses a **3-2-1 backup strategy**:
- **3** copies of data (production + 2 backups)
- **2** different storage types (local + cloud)
- **1** off-site backup

### What to Backup

#### Critical Data (Daily Backups)

1. **SQLite Database** (`data/trades.db`):
   - Contains all trade history
   - Position records
   - System settings
   - Signal explanations

2. **Parquet Cache** (`data/cache/*.parquet`):
   - Historical market data
   - Can be regenerated but saves API calls

3. **Configuration Files**:
   - `.env` (API keys, broker credentials)
   - Custom strategy parameters
   - Risk management settings

4. **Logs**:
   - Error logs
   - Trade execution logs
   - System event logs

### Automated Backup Script

Create `/opt/fluxhero/scripts/backup_all.sh`:

```bash
#!/bin/bash
# FluxHero Daily Backup Script
# Run via cron: 0 2 * * * /opt/fluxhero/scripts/backup_all.sh

set -e  # Exit on error

# Configuration
FLUXHERO_DIR="/opt/fluxhero"
BACKUP_DIR="/opt/fluxhero/backups"
DATA_DIR="${FLUXHERO_DIR}/data"
LOG_DIR="${FLUXHERO_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="fluxhero_backup_${TIMESTAMP}"
RETENTION_DAYS=30

# Cloud backup settings (configure for your provider)
CLOUD_BACKUP_ENABLED=true
S3_BUCKET="fluxhero-backups"  # Or your cloud storage
REMOTE_BACKUP_DIR="/mnt/backup"  # NFS/remote mount

echo "[$(date)] Starting FluxHero backup..."

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup SQLite database
echo "[$(date)] Backing up SQLite database..."
if [ -f "${DATA_DIR}/trades.db" ]; then
    sqlite3 "${DATA_DIR}/trades.db" ".backup '${BACKUP_DIR}/${BACKUP_NAME}/trades.db'"
    echo "[$(date)] SQLite backup complete"
else
    echo "[$(date)] WARNING: trades.db not found"
fi

# Backup Parquet cache (optional - can regenerate)
echo "[$(date)] Backing up Parquet cache..."
if [ -d "${DATA_DIR}/cache" ]; then
    cp -r "${DATA_DIR}/cache" "${BACKUP_DIR}/${BACKUP_NAME}/"
    echo "[$(date)] Parquet cache backup complete"
fi

# Backup configuration
echo "[$(date)] Backing up configuration..."
if [ -f "${FLUXHERO_DIR}/.env" ]; then
    cp "${FLUXHERO_DIR}/.env" "${BACKUP_DIR}/${BACKUP_NAME}/"
fi

# Backup logs (last 7 days)
echo "[$(date)] Backing up recent logs..."
if [ -d "${LOG_DIR}" ]; then
    find "${LOG_DIR}" -name "*.log" -mtime -7 -exec cp {} "${BACKUP_DIR}/${BACKUP_NAME}/" \;
fi

# Create compressed archive
echo "[$(date)] Creating compressed archive..."
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
rm -rf "${BACKUP_NAME}"

# Calculate checksum
sha256sum "${BACKUP_NAME}.tar.gz" > "${BACKUP_NAME}.tar.gz.sha256"

echo "[$(date)] Local backup complete: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"

# Upload to cloud storage (if enabled)
if [ "$CLOUD_BACKUP_ENABLED" = true ]; then
    echo "[$(date)] Uploading to cloud storage..."

    # AWS S3 example
    # aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" "s3://${S3_BUCKET}/"

    # Or rsync to remote server
    if [ -d "$REMOTE_BACKUP_DIR" ]; then
        rsync -avz "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" "${REMOTE_BACKUP_DIR}/"
        echo "[$(date)] Cloud backup complete"
    fi
fi

# Cleanup old backups (keep last 30 days)
echo "[$(date)] Cleaning up old backups..."
find "${BACKUP_DIR}" -name "fluxhero_backup_*.tar.gz" -mtime +${RETENTION_DAYS} -delete
find "${BACKUP_DIR}" -name "fluxhero_backup_*.sha256" -mtime +${RETENTION_DAYS} -delete

# Verify backup integrity
echo "[$(date)] Verifying backup integrity..."
if sha256sum -c "${BACKUP_NAME}.tar.gz.sha256" > /dev/null 2>&1; then
    echo "[$(date)] Backup integrity verified"
else
    echo "[$(date)] ERROR: Backup integrity check failed!"
    exit 1
fi

echo "[$(date)] Backup complete!"

# Send notification (optional)
# curl -X POST https://your-notification-service.com/notify \
#   -d "message=FluxHero backup completed successfully: ${BACKUP_NAME}"
```

Make script executable:
```bash
chmod +x /opt/fluxhero/scripts/backup_all.sh
```

### Schedule Automated Backups

Add to crontab (`crontab -e`):

```bash
# Daily backup at 2 AM
0 2 * * * /opt/fluxhero/scripts/backup_all.sh >> /opt/fluxhero/logs/backup.log 2>&1

# Weekly full system backup (Sundays at 3 AM)
0 3 * * 0 /opt/fluxhero/scripts/backup_full_system.sh >> /opt/fluxhero/logs/backup.log 2>&1
```

### Restore Procedure

#### Restore from Local Backup

```bash
#!/bin/bash
# Restore script
BACKUP_FILE="$1"
FLUXHERO_DIR="/opt/fluxhero"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Stop services
echo "Stopping FluxHero services..."
sudo systemctl stop fluxhero-backend
sudo systemctl stop fluxhero-frontend

# Create restore point of current state
echo "Creating restore point of current state..."
RESTORE_POINT="/opt/fluxhero/backups/pre_restore_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESTORE_POINT"
cp -r "${FLUXHERO_DIR}/data" "$RESTORE_POINT/"

# Extract backup
echo "Extracting backup..."
TEMP_DIR=$(mktemp -d)
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"

# Restore database
BACKUP_DIR=$(find "$TEMP_DIR" -name "fluxhero_backup_*" -type d | head -1)
if [ -f "${BACKUP_DIR}/trades.db" ]; then
    echo "Restoring database..."
    cp "${BACKUP_DIR}/trades.db" "${FLUXHERO_DIR}/data/"
fi

# Restore cache
if [ -d "${BACKUP_DIR}/cache" ]; then
    echo "Restoring cache..."
    cp -r "${BACKUP_DIR}/cache" "${FLUXHERO_DIR}/data/"
fi

# Restore configuration
if [ -f "${BACKUP_DIR}/.env" ]; then
    echo "Restoring configuration..."
    cp "${BACKUP_DIR}/.env" "$FLUXHERO_DIR/"
fi

# Cleanup
rm -rf "$TEMP_DIR"

# Restart services
echo "Restarting FluxHero services..."
sudo systemctl start fluxhero-backend
sudo systemctl start fluxhero-frontend

# Verify
sleep 5
curl http://localhost:8000/api/status

echo "Restore complete!"
echo "Restore point saved at: $RESTORE_POINT"
```

### Backup Testing

**Monthly test** (first week of each month):

```bash
# 1. Perform test restore on staging server
scp /opt/fluxhero/backups/latest.tar.gz user@staging:/tmp/

# 2. Restore on staging
ssh user@staging
cd /opt/fluxhero
./scripts/restore_backup.sh /tmp/latest.tar.gz

# 3. Verify data integrity
python -c "import sqlite3; conn = sqlite3.connect('/opt/fluxhero/data/trades.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM trades'); print(f'Trade count: {cursor.fetchone()[0]}'); conn.close()"

# 4. Run system health check
curl http://staging:8000/api/status
```

### Database-Specific Backups

#### SQLite Backup (Hot Backup)

```bash
# Backup while database is in use
sqlite3 /opt/fluxhero/data/trades.db "
.timeout 2000
.backup '/opt/fluxhero/backups/trades_$(date +%Y%m%d_%H%M%S).db'
"
```

#### Export Trade History to CSV

```bash
# Monthly export for external archival
sqlite3 -header -csv /opt/fluxhero/data/trades.db "
SELECT * FROM trades
WHERE timestamp >= date('now', '-30 days');
" > /opt/fluxhero/exports/trades_$(date +%Y%m).csv
```

---

## System Monitoring

### Key Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| System uptime | 99.9% | <99% |
| API response time | <100ms | >500ms |
| WebSocket latency | <50ms | >200ms |
| CPU usage | <50% | >80% |
| Memory usage | <70% | >85% |
| Disk usage | <80% | >90% |
| Trade execution lag | <2s | >10s |
| Daily drawdown | <3% | >5% |

### Monitoring Tools Setup

#### 1. System Resource Monitoring

Install `htop` and `glances`:

```bash
sudo apt-get install htop glances

# Run glances in web server mode
glances -w --port 61208
# Access at http://your-server:61208
```

#### 2. Application Health Monitoring

Create `/opt/fluxhero/scripts/health_check.sh`:

```bash
#!/bin/bash
# FluxHero Health Check Script
# Run every 5 minutes via cron

set -e

LOG_FILE="/opt/fluxhero/logs/health_check.log"
ALERT_FILE="/opt/fluxhero/logs/alerts.log"
API_URL="http://localhost:8000/api/status"
WEBSOCKET_URL="ws://localhost:8000/ws/prices"

echo "[$(date)] Starting health check..." >> "$LOG_FILE"

# Check backend API
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL")
if [ "$HTTP_CODE" != "200" ]; then
    echo "[$(date)] ALERT: API is down (HTTP $HTTP_CODE)" >> "$ALERT_FILE"
    # Send notification
    # curl -X POST https://your-notification-service.com/alert -d "Backend API is down"
fi

# Check database
DB_PATH="/opt/fluxhero/data/trades.db"
if ! sqlite3 "$DB_PATH" "SELECT 1;" > /dev/null 2>&1; then
    echo "[$(date)] ALERT: Database is inaccessible" >> "$ALERT_FILE"
fi

# Check disk space
DISK_USAGE=$(df /opt/fluxhero | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "[$(date)] ALERT: Disk usage at ${DISK_USAGE}%" >> "$ALERT_FILE"
fi

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100}')
if [ "$MEM_USAGE" -gt 85 ]; then
    echo "[$(date)] ALERT: Memory usage at ${MEM_USAGE}%" >> "$ALERT_FILE"
fi

# Check process status
if ! pgrep -f "uvicorn.*fluxhero" > /dev/null; then
    echo "[$(date)] ALERT: Backend process not running" >> "$ALERT_FILE"
    # Auto-restart
    sudo systemctl restart fluxhero-backend
fi

# Check log for recent errors
ERROR_COUNT=$(grep -c "ERROR" /opt/fluxhero/logs/fluxhero.log 2>/dev/null || echo 0)
if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "[$(date)] WARNING: ${ERROR_COUNT} errors in log file" >> "$ALERT_FILE"
fi

echo "[$(date)] Health check complete" >> "$LOG_FILE"
```

Schedule in crontab:
```bash
# Health check every 5 minutes
*/5 * * * * /opt/fluxhero/scripts/health_check.sh
```

#### 3. Trading-Specific Monitoring

Create `/opt/fluxhero/scripts/trading_monitor.sh`:

```bash
#!/bin/bash
# Monitor trading metrics
# Run every hour during market hours

DB_PATH="/opt/fluxhero/data/trades.db"
LOG_FILE="/opt/fluxhero/logs/trading_monitor.log"

echo "[$(date)] Trading metrics check..." >> "$LOG_FILE"

# Check daily drawdown
DAILY_DD=$(sqlite3 "$DB_PATH" "
SELECT ROUND((SUM(pnl) / (SELECT SUM(pnl) FROM trades WHERE pnl > 0 AND DATE(timestamp) = DATE('now')) * -100), 2)
FROM trades
WHERE pnl < 0 AND DATE(timestamp) = DATE('now');
")

if [ $(echo "$DAILY_DD > 3.0" | bc) -eq 1 ]; then
    echo "[$(date)] ALERT: Daily drawdown at ${DAILY_DD}%" >> "$LOG_FILE"
fi

# Check trade count (should have some activity)
TRADE_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = DATE('now');")
echo "[$(date)] Trades today: $TRADE_COUNT" >> "$LOG_FILE"

# Check for stuck positions (open >24 hours)
STUCK_POSITIONS=$(sqlite3 "$DB_PATH" "
SELECT COUNT(*) FROM positions
WHERE status = 'open'
AND datetime(opened_at) < datetime('now', '-1 day');
")

if [ "$STUCK_POSITIONS" -gt 0 ]; then
    echo "[$(date)] WARNING: $STUCK_POSITIONS positions open >24h" >> "$LOG_FILE"
fi
```

#### 4. Performance Benchmarking

Create monthly performance test:

```bash
#!/bin/bash
# Monthly performance benchmark
# Ensures computation speed hasn't degraded

cd /opt/fluxhero
source venv/bin/activate

python << 'EOF'
import time
import numpy as np
from fluxhero.backend.computation.indicators import calculate_ema, calculate_rsi
from fluxhero.backend.computation.adaptive_ema import calculate_kama

# Generate test data
np.random.seed(42)
prices = np.random.randn(10000).cumsum() + 100

# Benchmark EMA
start = time.perf_counter()
for _ in range(100):
    calculate_ema(prices, 20)
ema_time = (time.perf_counter() - start) * 1000

# Benchmark RSI
start = time.perf_counter()
for _ in range(100):
    calculate_rsi(prices, 14)
rsi_time = (time.perf_counter() - start) * 1000

# Benchmark KAMA
start = time.perf_counter()
for _ in range(100):
    calculate_kama(prices, 10, 2, 30)
kama_time = (time.perf_counter() - start) * 1000

print(f"EMA (100 iterations): {ema_time:.2f}ms")
print(f"RSI (100 iterations): {rsi_time:.2f}ms")
print(f"KAMA (100 iterations): {kama_time:.2f}ms")

# Alert if performance degraded
if ema_time > 50:  # Should be <50ms for 100 iterations
    print(f"ALERT: EMA performance degraded ({ema_time:.2f}ms)")
if rsi_time > 100:
    print(f"ALERT: RSI performance degraded ({rsi_time:.2f}ms)")
if kama_time > 150:
    print(f"ALERT: KAMA performance degraded ({kama_time:.2f}ms)")
EOF
```

### Alert Notifications

Configure alerts via webhook/email:

```bash
# Add to health_check.sh
send_alert() {
    MESSAGE="$1"

    # Slack webhook example
    curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
        -H 'Content-Type: application/json' \
        -d "{\"text\":\"FluxHero Alert: $MESSAGE\"}"

    # Or email via sendmail
    echo "$MESSAGE" | mail -s "FluxHero Alert" your-email@example.com
}
```

---

## Database Maintenance

### SQLite Optimization

#### Weekly VACUUM (Sundays at 3 AM)

```bash
#!/bin/bash
# Optimize SQLite database
# Run weekly via cron: 0 3 * * 0 /opt/fluxhero/scripts/db_vacuum.sh

DB_PATH="/opt/fluxhero/data/trades.db"

echo "[$(date)] Starting database optimization..."

# Backup before VACUUM
sqlite3 "$DB_PATH" ".backup '${DB_PATH}.pre_vacuum'"

# VACUUM to reclaim space and optimize
sqlite3 "$DB_PATH" "VACUUM;"

# Analyze to update statistics
sqlite3 "$DB_PATH" "ANALYZE;"

# Check integrity
INTEGRITY=$(sqlite3 "$DB_PATH" "PRAGMA integrity_check;")
if [ "$INTEGRITY" = "ok" ]; then
    echo "[$(date)] Database optimization complete and verified"
    rm "${DB_PATH}.pre_vacuum"
else
    echo "[$(date)] ERROR: Integrity check failed: $INTEGRITY"
    # Restore backup
    mv "${DB_PATH}.pre_vacuum" "$DB_PATH"
fi
```

#### Monthly Archive (Archive old trades)

```bash
#!/bin/bash
# Archive trades older than 90 days
# Run monthly: 0 4 1 * * /opt/fluxhero/scripts/archive_old_trades.sh

DB_PATH="/opt/fluxhero/data/trades.db"
ARCHIVE_PATH="/opt/fluxhero/data/archive/trades_archive_$(date +%Y%m).db"

mkdir -p "$(dirname "$ARCHIVE_PATH")"

# Create archive database
sqlite3 "$ARCHIVE_PATH" < /opt/fluxhero/backend/storage/schema.sql

# Copy old trades to archive
sqlite3 "$DB_PATH" "
ATTACH DATABASE '$ARCHIVE_PATH' AS archive;
INSERT INTO archive.trades
SELECT * FROM main.trades
WHERE datetime(timestamp) < datetime('now', '-90 days');
DETACH DATABASE archive;
"

# Delete archived trades from main database (optional)
# Uncomment if you want to remove old trades
# sqlite3 "$DB_PATH" "DELETE FROM trades WHERE datetime(timestamp) < datetime('now', '-90 days');"

echo "[$(date)] Archived trades to $ARCHIVE_PATH"
```

### Database Size Monitoring

```bash
# Check database size
du -h /opt/fluxhero/data/trades.db

# Check table sizes
sqlite3 /opt/fluxhero/data/trades.db "
SELECT
    name,
    SUM(pgsize) as size_bytes,
    ROUND(SUM(pgsize) / 1024.0 / 1024.0, 2) as size_mb
FROM dbstat
GROUP BY name
ORDER BY size_bytes DESC;
"
```

---

## Log Management

### Log Rotation

Create `/etc/logrotate.d/fluxhero`:

```
/opt/fluxhero/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 fluxhero fluxhero
    sharedscripts
    postrotate
        systemctl reload fluxhero-backend > /dev/null 2>&1 || true
    endscript
}
```

### Log Analysis

Weekly log analysis script:

```bash
#!/bin/bash
# Analyze logs for errors and warnings
# Run weekly: 0 5 * * 1 /opt/fluxhero/scripts/analyze_logs.sh

LOG_DIR="/opt/fluxhero/logs"
REPORT_FILE="/opt/fluxhero/logs/weekly_report_$(date +%Y%m%d).txt"

echo "FluxHero Weekly Log Analysis - $(date)" > "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"

# Count errors by type
echo -e "\nError Summary (Last 7 Days):" >> "$REPORT_FILE"
grep "ERROR" "$LOG_DIR/fluxhero.log" | \
    awk '{print $4}' | sort | uniq -c | sort -rn >> "$REPORT_FILE"

# Count warnings
echo -e "\nWarning Count:" >> "$REPORT_FILE"
grep -c "WARNING" "$LOG_DIR/fluxhero.log" >> "$REPORT_FILE"

# Failed trades
echo -e "\nFailed Trades:" >> "$REPORT_FILE"
grep "Trade execution failed" "$LOG_DIR/fluxhero.log" | wc -l >> "$REPORT_FILE"

# API errors
echo -e "\nAPI Errors:" >> "$REPORT_FILE"
grep "API request failed" "$LOG_DIR/fluxhero.log" | wc -l >> "$REPORT_FILE"

# Top error messages
echo -e "\nTop 10 Error Messages:" >> "$REPORT_FILE"
grep "ERROR" "$LOG_DIR/fluxhero.log" | \
    sed 's/.*ERROR/ERROR/' | sort | uniq -c | sort -rn | head -10 >> "$REPORT_FILE"

echo -e "\nReport saved to: $REPORT_FILE"
```

---

## Performance Optimization

### Monthly Performance Audit

```bash
#!/bin/bash
# Monthly performance audit
# Check for performance degradation

cd /opt/fluxhero
source venv/bin/activate

# Run pytest benchmarks
pytest tests/computation/ --benchmark-only --benchmark-json=benchmark_$(date +%Y%m).json

# Compare with previous month
python << 'EOF'
import json
from datetime import datetime, timedelta

current_month = datetime.now().strftime('%Y%m')
previous_month = (datetime.now() - timedelta(days=30)).strftime('%Y%m')

try:
    with open(f'benchmark_{current_month}.json') as f:
        current = json.load(f)
    with open(f'benchmark_{previous_month}.json') as f:
        previous = json.load(f)

    print("Performance Comparison:")
    # Compare benchmark results
    # (Implementation depends on benchmark structure)
except FileNotFoundError:
    print("Previous benchmark not found - first run")
EOF
```

### Cache Cleanup

```bash
#!/bin/bash
# Clean old cache files
# Run weekly: 0 4 * * 0 /opt/fluxhero/scripts/clean_cache.sh

CACHE_DIR="/opt/fluxhero/data/cache"
RETENTION_DAYS=30

echo "[$(date)] Cleaning cache older than $RETENTION_DAYS days..."

# Remove old parquet files
find "$CACHE_DIR" -name "*.parquet" -mtime +$RETENTION_DAYS -delete

# Remove empty directories
find "$CACHE_DIR" -type d -empty -delete

echo "[$(date)] Cache cleanup complete"
```

---

## Security Updates

### Security Update Policy

- **Critical vulnerabilities**: Apply within 24 hours
- **High severity**: Apply within 1 week
- **Medium severity**: Apply during monthly update cycle
- **Low severity**: Apply during quarterly maintenance

### Security Monitoring

Subscribe to security advisories:

1. **Python Security**: https://www.python.org/news/security/
2. **npm advisories**: Run `npm audit` weekly
3. **Ubuntu Security**: `sudo apt list --upgradable | grep security`

### Security Update Procedure

```bash
#!/bin/bash
# Apply security updates
# Run as needed for critical patches

echo "Checking for security updates..."

# Python packages
pip-audit --fix

# System packages
sudo apt-get update
sudo apt-get upgrade -y

# Restart services
sudo systemctl restart fluxhero-backend
sudo systemctl restart fluxhero-frontend

echo "Security updates applied"
```

---

## Disaster Recovery

### Disaster Recovery Plan

#### Recovery Time Objectives (RTO)

| Scenario | Target RTO | Steps |
|----------|-----------|-------|
| Server failure | 2 hours | Restore from backup to new server |
| Database corruption | 1 hour | Restore from latest backup |
| Data loss | 30 minutes | Restore from daily backup |
| Configuration loss | 15 minutes | Restore .env from backup |

#### Recovery Point Objectives (RPO)

- **Trade data**: 24 hours (daily backups)
- **Market data**: Acceptable to regenerate
- **Configuration**: 24 hours

### Full System Recovery Procedure

```bash
# 1. Provision new server (if needed)
# 2. Install dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv nodejs npm sqlite3

# 3. Clone repository or copy files
git clone https://github.com/your-repo/fluxhero.git /opt/fluxhero
# OR
scp -r backup-server:/opt/fluxhero /opt/

# 4. Restore data from backup
cd /opt/fluxhero
./scripts/restore_backup.sh /path/to/latest_backup.tar.gz

# 5. Install Python dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Install Node dependencies
cd fluxhero/frontend
npm ci
npm run build

# 7. Configure services
sudo cp /opt/fluxhero/deployment/systemd/* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable fluxhero-backend fluxhero-frontend
sudo systemctl start fluxhero-backend fluxhero-frontend

# 8. Verify system
curl http://localhost:8000/api/status
curl http://localhost:3000

# 9. Update DNS (if needed)
# Point domain to new server IP
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: High Memory Usage

**Diagnosis**:
```bash
# Check process memory
ps aux | grep fluxhero | awk '{print $2, $4, $11}'

# Check Python memory profiling
python -m memory_profiler backend/main.py
```

**Solutions**:
- Reduce candle buffer size (default: 500)
- Clear parquet cache: `rm -rf data/cache/*`
- Restart services: `sudo systemctl restart fluxhero-backend`

#### Issue: Slow API Response

**Diagnosis**:
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/positions
```

**Solutions**:
- Check database size: `du -h data/trades.db`
- Run VACUUM: `sqlite3 data/trades.db "VACUUM;"`
- Add database indexes (if missing)

#### Issue: WebSocket Disconnections

**Diagnosis**:
```bash
# Check WebSocket logs
grep "WebSocket" logs/fluxhero.log | tail -20
```

**Solutions**:
- Check network connectivity
- Verify broker API status
- Increase reconnection timeout
- Check firewall rules

#### Issue: Backtests Producing Different Results

**Diagnosis**:
- Check for dependency version changes
- Verify data integrity
- Check for random seed issues

**Solutions**:
```bash
# Compare dependency versions
pip freeze > current_deps.txt
diff current_deps.txt known_good_deps.txt

# Restore specific package version
pip install pandas==2.2.0  # Example
```

### Emergency Procedures

#### Emergency Stop (Kill Switch)

```bash
# Stop all trading immediately
curl -X POST http://localhost:8000/api/emergency_stop

# Verify all positions closed
curl http://localhost:8000/api/positions
```

#### Database Recovery

```bash
# If database is corrupted
cd /opt/fluxhero/data
mv trades.db trades.db.corrupted
sqlite3 trades.db.corrupted ".recover" | sqlite3 trades.db
```

---

## Maintenance Schedules

### Daily Tasks (Automated)

- **2:00 AM**: Full system backup
- **Every 5 min**: Health check
- **Every hour**: Trading metrics monitoring
- **Daily**: Log rotation

### Weekly Tasks (Automated)

- **Sunday 3:00 AM**: Database VACUUM
- **Sunday 4:00 AM**: Cache cleanup
- **Monday 5:00 AM**: Log analysis report

### Monthly Tasks (Manual)

- **First week**: Dependency updates (following update procedure)
- **First week**: Backup restoration test
- **Mid-month**: Performance audit
- **End of month**: Security review

### Quarterly Tasks (Manual)

- **Performance tuning**: Review and optimize slow queries
- **Capacity planning**: Review disk/memory usage trends
- **Disaster recovery drill**: Full system recovery test
- **Documentation update**: Update this guide with new procedures

### Maintenance Log Template

Keep a maintenance log at `/opt/fluxhero/docs/maintenance_log.md`:

```markdown
# FluxHero Maintenance Log

## 2026-02-01: Monthly Update
- Updated Python dependencies (pandas 2.2.0 → 2.2.1)
- Updated Node dependencies (next 16.1.4 → 16.2.0)
- Ran full test suite: PASSED
- Deployed to production: 2026-02-01 10:00 UTC
- Issues: None
- Rollback required: No

## 2026-01-28: Database Maintenance
- Ran VACUUM on trades.db
- Database size before: 2.4 GB
- Database size after: 2.1 GB
- Reclaimed: 300 MB
- Integrity check: PASSED

## 2026-01-21: Security Patch
- Applied critical security patch for httpx
- Version: httpx 0.27.0 → 0.27.2
- CVE: CVE-2026-XXXXX
- Impact: Fixed potential DoS vulnerability
- Testing: Staging tested for 4 hours
- Deployed: 2026-01-21 16:30 UTC
```

---

## Summary

### Maintenance Checklist

#### Weekly Checklist
- [ ] Review health check logs
- [ ] Check backup success status
- [ ] Review trading performance metrics
- [ ] Check disk space usage
- [ ] Review error logs

#### Monthly Checklist
- [ ] Update dependencies (following procedure)
- [ ] Test backup restoration
- [ ] Run performance benchmarks
- [ ] Review and rotate old logs
- [ ] Update maintenance log
- [ ] Review security advisories

#### Quarterly Checklist
- [ ] Full disaster recovery drill
- [ ] Capacity planning review
- [ ] Performance optimization audit
- [ ] Update documentation
- [ ] Review and update monitoring thresholds

### Key Commands Reference

```bash
# Backup
/opt/fluxhero/scripts/backup_all.sh

# Restore
/opt/fluxhero/scripts/restore_backup.sh /path/to/backup.tar.gz

# Health check
/opt/fluxhero/scripts/health_check.sh

# Database optimization
sqlite3 /opt/fluxhero/data/trades.db "VACUUM; ANALYZE;"

# Update dependencies
pip list --outdated
npm outdated

# View logs
tail -f /opt/fluxhero/logs/fluxhero.log

# Check service status
sudo systemctl status fluxhero-backend
sudo systemctl status fluxhero-frontend

# Restart services
sudo systemctl restart fluxhero-backend
sudo systemctl restart fluxhero-frontend
```

---

## Support and Resources

- **Documentation**: `/opt/fluxhero/docs/`
- **Logs**: `/opt/fluxhero/logs/`
- **Backups**: `/opt/fluxhero/backups/`
- **Issue Tracker**: [Your GitHub repo]
- **Email Support**: [Your support email]

---

**Document Version**: 1.0.0
**Last Updated**: 2026-01-21
**Next Review**: 2026-04-21
