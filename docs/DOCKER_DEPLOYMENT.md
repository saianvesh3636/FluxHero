# FluxHero Docker Deployment Guide

**Version**: 1.0.0
**Last Updated**: 2026-01-23
**Target**: Docker-based development and production deployment

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Build and Run](#build-and-run)
6. [Volume Mounts](#volume-mounts)
7. [Production Deployment](#production-deployment)
8. [Nginx Reverse Proxy](#nginx-reverse-proxy)
9. [SSL/TLS Configuration](#ssltls-configuration)
10. [Environment Configuration](#environment-configuration)
11. [Makefile Commands](#makefile-commands)
12. [Troubleshooting](#troubleshooting)

---

## Overview

FluxHero provides Docker containers for both the backend (FastAPI) and frontend (Next.js) services. The Docker setup uses:

- **Backend**: Python 3.11-slim with uv package manager
- **Frontend**: Node 20-alpine with multi-stage build
- **Orchestration**: Docker Compose for service management
- **Networking**: Bridge network for inter-service communication

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Host                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 fluxhero-network                         │ │
│  │  ┌──────────────────────┐   ┌──────────────────────┐    │ │
│  │  │  fluxhero-backend    │   │  fluxhero-frontend   │    │ │
│  │  │  (FastAPI + Uvicorn) │◄──│  (Next.js)           │    │ │
│  │  │  Port: 8000          │   │  Port: 3000          │    │ │
│  │  └──────────────────────┘   └──────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                               │
│  ┌───────────────────────────┴───────────────────────────┐   │
│  │              Volume Mounts                             │   │
│  │  ./data → /app/data (SQLite DB, parquet cache)        │   │
│  │  ./logs → /app/logs (application logs)                │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
              │                    │
         Port 8000            Port 3000
              │                    │
              └────────┬───────────┘
                       ▼
               Your Browser / API
```

---

## Prerequisites

Before deploying with Docker, ensure you have:

- **Docker Engine**: Version 20.10 or later
- **Docker Compose**: Version 2.0 or later (included with Docker Desktop)
- **Alpaca API credentials**: For broker connectivity
- **Sufficient disk space**: At least 5GB for images and data

### Verify Docker Installation

```bash
# Check Docker version
docker --version
# Expected: Docker version 20.10.x or later

# Check Docker Compose version
docker compose version
# Expected: Docker Compose version v2.x.x

# Verify Docker is running
docker ps
```

---

## Quick Start

Get FluxHero running in Docker with these steps:

```bash
# 1. Clone the repository (if not already done)
git clone https://github.com/your-username/fluxhero.git
cd fluxhero

# 2. Create environment file from template
cp docker/.env.docker.example .env

# 3. Edit .env with your credentials
# IMPORTANT: Set FLUXHERO_ENCRYPTION_KEY and Alpaca credentials
nano .env  # or use your preferred editor

# 4. Build and start services
make docker-build
make docker-up

# 5. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## Configuration

### Environment File Setup

Copy the Docker environment template and configure required variables:

```bash
cp docker/.env.docker.example .env
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `FLUXHERO_AUTH_SECRET` | Secret key for API authentication | `your-secure-random-string` |
| `FLUXHERO_ENCRYPTION_KEY` | AES-256 key for credential encryption | 64 hex characters |
| `FLUXHERO_ALPACA_API_KEY` | Alpaca API key | `AKXXXXXXXXXXXXXXXXXX` |
| `FLUXHERO_ALPACA_API_SECRET` | Alpaca API secret | `your-alpaca-secret` |

### Generate Secure Keys

```bash
# Generate authentication secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate encryption key (64 hex characters for AES-256)
python -c "import secrets; print(secrets.token_hex(32))"
```

### Environment Modes

**Development (Paper Trading)**:
```bash
FLUXHERO_ALPACA_API_URL=https://paper-api.alpaca.markets
```

**Production (Live Trading)**:
```bash
FLUXHERO_ALPACA_API_URL=https://api.alpaca.markets
FLUXHERO_ENV=production
```

---

## Build and Run

### Build Docker Images

```bash
# Build all images
make docker-build

# Or using docker compose directly
docker compose build

# Build with no cache (useful after dependency updates)
docker compose build --no-cache
```

### Start Services

```bash
# Start in detached mode (background)
make docker-up

# Or using docker compose directly
docker compose up -d

# Start and follow logs
docker compose up
```

### Stop Services

```bash
# Stop containers (preserve data)
make docker-down

# Stop and remove volumes (WARNING: deletes data)
make docker-clean
```

### View Logs

```bash
# Follow logs from all services
make docker-logs

# Follow logs from specific service
docker compose logs -f backend
docker compose logs -f frontend
```

---

## Volume Mounts

Docker volumes persist data between container restarts:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | SQLite database, parquet cache |
| `./logs` | `/app/logs` | Application log files |

### Data Directory Structure

```
./data/
├── fluxhero.db          # SQLite database
├── cache/               # Parquet market data cache
│   └── *.parquet        # Cached historical data
└── paper_trades.db      # Paper trading state

./logs/
└── fluxhero.log         # Application logs
```

### Backup Data

```bash
# Backup SQLite database
cp ./data/fluxhero.db ./data/fluxhero.db.bak

# Backup entire data directory
tar -czvf fluxhero-backup-$(date +%Y%m%d).tar.gz ./data ./logs
```

### Restore Data

```bash
# Restore from backup
tar -xzvf fluxhero-backup-YYYYMMDD.tar.gz

# Restart containers to pick up restored data
make docker-down && make docker-up
```

---

## Production Deployment

### Production Environment Variables

Set these additional variables for production:

```bash
# Set production mode
FLUXHERO_ENV=production

# Use live Alpaca API
FLUXHERO_ALPACA_API_URL=https://api.alpaca.markets

# Ensure strong secrets are set
FLUXHERO_AUTH_SECRET=<strong-production-secret>
FLUXHERO_ENCRYPTION_KEY=<secure-encryption-key>
```

### Production Checklist

- [ ] Strong `FLUXHERO_AUTH_SECRET` set (not default)
- [ ] Strong `FLUXHERO_ENCRYPTION_KEY` set (64 hex characters)
- [ ] Alpaca live credentials configured
- [ ] Nginx reverse proxy with SSL configured
- [ ] UFW firewall enabled (ports 80, 443 only)
- [ ] Data volume on persistent storage
- [ ] Automated backups configured
- [ ] Monitoring/alerting configured

### Docker Compose Production Overrides

For production, create a `docker-compose.prod.yml`:

```yaml
# docker-compose.prod.yml
services:
  backend:
    restart: always
    environment:
      - FLUXHERO_ENV=production
    deploy:
      resources:
        limits:
          memory: 2G

  frontend:
    restart: always
    deploy:
      resources:
        limits:
          memory: 1G
```

Run with production overrides:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Nginx Reverse Proxy

For production, use Nginx as a reverse proxy with SSL termination.

### Install Nginx

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y nginx
```

### Nginx Configuration

Create `/etc/nginx/sites-available/fluxhero`:

```nginx
# HTTP - Redirect to HTTPS
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS - Main server block
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL Configuration (managed by Certbot)
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Frontend (Next.js on port 3000)
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API (FastAPI on port 8000)
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket connections
    location /ws {
        proxy_pass http://localhost:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://localhost:8000/health;
    }
}
```

### Enable Configuration

```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/fluxhero /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

---

## SSL/TLS Configuration

### Using Let's Encrypt (Recommended)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificate (Certbot configures Nginx automatically)
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Verify auto-renewal is enabled
sudo certbot renew --dry-run
```

### Certificate Renewal

Certbot sets up automatic renewal via systemd timer:

```bash
# Check renewal timer status
sudo systemctl status certbot.timer

# Manual renewal (if needed)
sudo certbot renew
```

---

## Environment Configuration

### Development vs Production

| Setting | Development | Production |
|---------|-------------|------------|
| `FLUXHERO_ENV` | (not set) | `production` |
| `FLUXHERO_ALPACA_API_URL` | `https://paper-api.alpaca.markets` | `https://api.alpaca.markets` |
| SSL | Not required | Required |
| Restart policy | `unless-stopped` | `always` |

### CORS Configuration

For production with a custom domain, update CORS origins:

```bash
FLUXHERO_CORS_ORIGINS=["https://yourdomain.com","https://www.yourdomain.com"]
```

### Inter-Service Communication

The frontend communicates with the backend via Docker network:

- **Internal (Docker network)**: `http://backend:8000`
- **External (browser)**: Frontend proxies API calls

These are set in `docker-compose.yml`:

```yaml
frontend:
  environment:
    - NEXT_PUBLIC_API_URL=http://backend:8000
    - NEXT_PUBLIC_WS_URL=ws://backend:8000
```

---

## Makefile Commands

The Makefile provides convenient Docker commands:

| Command | Description |
|---------|-------------|
| `make docker-build` | Build Docker images |
| `make docker-up` | Start containers in detached mode |
| `make docker-down` | Stop and remove containers |
| `make docker-logs` | Follow container logs |
| `make docker-shell-backend` | Open bash shell in backend container |
| `make docker-clean` | Remove containers, volumes, and images |

### Examples

```bash
# Build and start
make docker-build && make docker-up

# Check running containers
docker compose ps

# Execute command in backend container
docker compose exec backend python -c "print('Hello from container')"

# View resource usage
docker stats
```

---

## Troubleshooting

### Issue: Containers won't start

**Check logs for errors:**
```bash
docker compose logs backend
docker compose logs frontend
```

**Verify environment file exists:**
```bash
ls -la .env
```

**Common fixes:**
- Ensure `.env` file is in project root
- Check that all required variables are set
- Verify no syntax errors in `.env` file

### Issue: Backend health check fails

**Check if backend is responding:**
```bash
curl http://localhost:8000/health
```

**Check backend logs:**
```bash
docker compose logs -f backend
```

**Common causes:**
- Missing environment variables
- Database connection issues
- Port conflict (8000 already in use)

### Issue: Frontend can't connect to backend

**Check Docker network:**
```bash
docker network ls
docker network inspect fluxhero-network
```

**Verify services are on same network:**
```bash
docker compose ps
```

**Test connectivity from frontend:**
```bash
docker compose exec frontend wget -qO- http://backend:8000/health
```

### Issue: Permission denied on volumes

**Fix volume permissions:**
```bash
# Create directories with correct permissions
mkdir -p data logs
chmod 755 data logs

# If running as non-root in container
sudo chown -R 1001:1001 data logs
```

### Issue: Container runs out of memory

**Increase Docker memory limit:**
- Docker Desktop: Settings → Resources → Memory
- Linux: Configure in `/etc/docker/daemon.json`

**Add memory limits in docker-compose.yml:**
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
```

### Issue: SSL certificate issues

**Verify certificate paths:**
```bash
sudo ls -la /etc/letsencrypt/live/yourdomain.com/
```

**Test Nginx configuration:**
```bash
sudo nginx -t
```

**Check certificate expiry:**
```bash
sudo certbot certificates
```

### Issue: Database locked

**Stop containers and restart:**
```bash
make docker-down
sleep 5
make docker-up
```

**Check for stale lock files:**
```bash
ls -la ./data/*.db*
```

### Issue: Slow container startup

**Causes:**
- Large Docker images (first build)
- Slow health check intervals
- Network initialization

**Solutions:**
- Use `docker compose pull` to pre-fetch images
- Increase `start_period` in health checks
- Build images in advance: `make docker-build`

---

## Related Documentation

- [Deployment Guide](DEPLOYMENT_GUIDE.md) - VPS/Cloud deployment without Docker
- [Maintenance Guide](MAINTENANCE_GUIDE.md) - Ongoing maintenance tasks
- [API Documentation](API_DOCUMENTATION.md) - API reference
- [User Guide](USER_GUIDE.md) - Application usage

---

**Last Updated**: 2026-01-23
**Version**: 1.0.0
