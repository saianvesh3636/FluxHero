# FluxHero Deployment Guide

**Version**: 1.0.0
**Last Updated**: 2026-01-21
**Target**: VPS/Cloud Production Deployment

---

## Table of Contents

1. [Overview](#overview)
2. [Hosting Options Comparison](#hosting-options-comparison)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [Deployment Option A: AWS EC2](#deployment-option-a-aws-ec2)
5. [Deployment Option B: DigitalOcean Droplet](#deployment-option-b-digitalocean-droplet)
6. [Deployment Option C: Linode/Akamai](#deployment-option-c-linodeakamai)
7. [Deployment Option D: Hetzner Cloud](#deployment-option-d-hetzner-cloud)
8. [Common Setup Steps](#common-setup-steps)
9. [Security Hardening](#security-hardening)
10. [Process Management](#process-management)
11. [SSL/TLS Configuration](#ssltls-configuration)
12. [Monitoring & Alerting](#monitoring--alerting)
13. [Backup & Disaster Recovery](#backup--disaster-recovery)
14. [Cost Optimization](#cost-optimization)
15. [Troubleshooting](#troubleshooting)

---

## Overview

FluxHero can be deployed on any Linux VPS or cloud instance. This guide covers:

- **Multiple cloud provider options** with cost comparisons
- **Step-by-step deployment instructions** for each provider
- **Security best practices** for production trading systems
- **Monitoring and alerting** setup
- **Backup strategies** for trade data and system state

### Recommended Deployment Architecture

```
┌─────────────────────────────────────────┐
│          Cloud VPS Instance             │
│  ┌───────────────────────────────────┐  │
│  │  FluxHero Backend (FastAPI)       │  │
│  │  - Port 8000 (internal)           │  │
│  │  - WebSocket feed                 │  │
│  │  - SQLite + Parquet storage       │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  FluxHero Frontend (Next.js)      │  │
│  │  - Port 3000 (internal)           │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  Nginx Reverse Proxy              │  │
│  │  - Port 80 (HTTP) → 443 (HTTPS)   │  │
│  │  - SSL termination                │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         │
         ▼
   Your Browser (HTTPS access)
```

---

## Hosting Options Comparison

### Option A: AWS EC2 (Amazon Web Services)

**Best For**: Enterprise users, those already in AWS ecosystem, need advanced features

| Metric | Details |
|--------|---------|
| **Recommended Instance** | t3.medium (2 vCPU, 4GB RAM) |
| **Cost** | ~$30-40/month (on-demand), ~$20/month (1-year reserved) |
| **Location** | Global (20+ regions), US East/West recommended for US markets |
| **Network** | Excellent, AWS backbone |
| **Storage** | EBS volumes (20GB GP3 included, ~$2/month) |
| **Pros** | Mature ecosystem, auto-scaling, CloudWatch monitoring, IAM security |
| **Cons** | More expensive, complex pricing, learning curve |

### Option B: DigitalOcean Droplet

**Best For**: Developers, simple setup, predictable pricing

| Metric | Details |
|--------|---------|
| **Recommended Droplet** | Basic 4GB RAM / 2 vCPU |
| **Cost** | $24/month (fixed pricing) |
| **Location** | 15 data centers, NYC/SF for US markets |
| **Network** | Very good, 4TB transfer included |
| **Storage** | 80GB SSD included |
| **Pros** | Simple pricing, excellent docs, managed databases available, easy UI |
| **Cons** | Fewer features than AWS, no spot instances |

### Option C: Linode (Akamai)

**Best For**: Cost-conscious users, good performance/price ratio

| Metric | Details |
|--------|---------|
| **Recommended Plan** | Linode 4GB (2 vCPU, 4GB RAM) |
| **Cost** | $24/month |
| **Location** | 11 global data centers, Atlanta/Newark for US East Coast |
| **Network** | Excellent (Akamai CDN backbone), 4TB transfer |
| **Storage** | 80GB SSD included |
| **Pros** | Great performance, competitive pricing, 24/7 support |
| **Cons** | Smaller ecosystem than AWS |

### Option D: Hetzner Cloud

**Best For**: Budget-conscious users in EU, best price/performance

| Metric | Details |
|--------|---------|
| **Recommended Plan** | CPX21 (3 vCPU, 4GB RAM) |
| **Cost** | ~$10-12/month (€9-11) |
| **Location** | EU only (Germany, Finland), **NOT ideal for US market data** |
| **Network** | Excellent in EU, higher latency to US |
| **Storage** | 80GB SSD included |
| **Pros** | Cheapest option, great EU performance |
| **Cons** | EU-only, higher latency to US broker APIs (200-300ms vs 10-50ms) |

### Recommendation Matrix

| Use Case | Recommended Provider | Reason |
|----------|---------------------|---------|
| **US-based trading, cost-sensitive** | DigitalOcean or Linode | Best price/performance, US data centers |
| **Enterprise, need advanced monitoring** | AWS EC2 | Mature tooling, CloudWatch, auto-scaling |
| **EU-based trading** | Hetzner Cloud | Best price, excellent EU performance |
| **Beginner-friendly** | DigitalOcean | Simplest UI, best documentation |

---

## Pre-Deployment Checklist

Before deploying, ensure you have:

- [ ] **Broker API credentials** (Alpaca, Interactive Brokers, etc.)
- [ ] **Domain name** (optional but recommended for HTTPS)
- [ ] **SSH key pair** generated (`ssh-keygen -t ed25519`)
- [ ] **Git repository access** (to pull FluxHero code)
- [ ] **Backup strategy planned** (see Backup section)
- [ ] **Monitoring plan** (email/SMS alerts for downtime)
- [ ] **Initial capital allocated** (recommend $10k+ for live trading)

---

## Deployment Option A: AWS EC2

### Step 1: Launch EC2 Instance

```bash
# Using AWS CLI (or use AWS Console)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 22.04 LTS (us-east-1)
  --instance-type t3.medium \
  --key-name your-keypair-name \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=FluxHero-Production}]'
```

**Via AWS Console**:
1. Go to EC2 → Launch Instance
2. Choose **Ubuntu 22.04 LTS** AMI
3. Select **t3.medium** instance type
4. Configure security group (see Security Hardening section)
5. Add 20GB GP3 EBS volume
6. Launch with your SSH key

### Step 2: Configure Security Group

Create security group with these inbound rules:

| Type | Port | Source | Purpose |
|------|------|--------|---------|
| SSH | 22 | Your IP only | SSH access |
| HTTP | 80 | 0.0.0.0/0 | Web access (redirect to HTTPS) |
| HTTPS | 443 | 0.0.0.0/0 | Secure web access |

```bash
# Create security group
aws ec2 create-security-group \
  --group-name fluxhero-sg \
  --description "FluxHero trading system"

# Add SSH rule (replace YOUR_IP)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp --port 22 --cidr YOUR_IP/32

# Add HTTP/HTTPS rules
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp --port 80 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp --port 443 --cidr 0.0.0.0/0
```

### Step 3: Connect and Setup

```bash
# Get instance public IP
aws ec2 describe-instances --instance-ids i-xxxxxxxxx \
  --query 'Reservations[0].Instances[0].PublicIpAddress'

# SSH into instance
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_INSTANCE_IP
```

**Continue to [Common Setup Steps](#common-setup-steps)**

---

## Deployment Option B: DigitalOcean Droplet

### Step 1: Create Droplet

**Via DigitalOcean Control Panel**:

1. Go to **Create** → **Droplets**
2. Choose image: **Ubuntu 22.04 LTS**
3. Choose plan: **Basic** → **$24/month (4GB RAM / 2 vCPU)**
4. Choose datacenter: **New York** or **San Francisco** (for US markets)
5. Add your SSH key
6. Hostname: `fluxhero-production`
7. Click **Create Droplet**

**Via doctl CLI**:

```bash
# Install doctl
brew install doctl  # macOS
# or
sudo snap install doctl  # Linux

# Authenticate
doctl auth init

# Create droplet
doctl compute droplet create fluxhero-production \
  --image ubuntu-22-04-x64 \
  --size s-2vcpu-4gb \
  --region nyc3 \
  --ssh-keys YOUR_SSH_KEY_ID \
  --tag-name production
```

### Step 2: Configure Firewall

```bash
# Create firewall
doctl compute firewall create \
  --name fluxhero-firewall \
  --inbound-rules "protocol:tcp,ports:22,sources:addresses:YOUR_IP/32 protocol:tcp,ports:80,sources:addresses:0.0.0.0/0 protocol:tcp,ports:443,sources:addresses:0.0.0.0/0" \
  --outbound-rules "protocol:tcp,ports:all,destinations:addresses:0.0.0.0/0 protocol:udp,ports:all,destinations:addresses:0.0.0.0/0"

# Apply to droplet
doctl compute firewall add-droplets FIREWALL_ID --droplet-ids DROPLET_ID
```

### Step 3: Connect

```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP
```

**Continue to [Common Setup Steps](#common-setup-steps)**

---

## Deployment Option C: Linode/Akamai

### Step 1: Create Linode

**Via Linode Cloud Manager**:

1. Go to **Create** → **Linode**
2. Choose distribution: **Ubuntu 22.04 LTS**
3. Choose region: **Newark, NJ** or **Atlanta, GA** (for US markets)
4. Choose plan: **Linode 4GB** ($24/month)
5. Add your SSH key
6. Label: `fluxhero-production`
7. Click **Create Linode**

**Via Linode CLI**:

```bash
# Install linode-cli
uv pip install linode-cli

# Configure
linode-cli configure

# Create instance
linode-cli linodes create \
  --type g6-standard-2 \
  --region us-east \
  --image linode/ubuntu22.04 \
  --label fluxhero-production \
  --root_pass YOUR_ROOT_PASSWORD \
  --authorized_keys "YOUR_SSH_PUBLIC_KEY"
```

### Step 2: Configure Firewall

```bash
# Create firewall rules via Cloud Manager or CLI
linode-cli firewalls create \
  --label fluxhero-firewall \
  --rules.inbound '[{"protocol":"TCP","ports":"22","addresses":{"ipv4":["YOUR_IP/32"]}}]' \
  --rules.inbound '[{"protocol":"TCP","ports":"80,443","addresses":{"ipv4":["0.0.0.0/0"]}}]'
```

### Step 3: Connect

```bash
ssh root@YOUR_LINODE_IP
```

**Continue to [Common Setup Steps](#common-setup-steps)**

---

## Deployment Option D: Hetzner Cloud

**Warning**: Only suitable for EU-based users. US market data latency will be 200-300ms vs 10-50ms on US servers.

### Step 1: Create Server

**Via Hetzner Cloud Console**:

1. Go to **Servers** → **Add Server**
2. Location: **Nuremberg** or **Helsinki**
3. Image: **Ubuntu 22.04**
4. Type: **CPX21** (3 vCPU, 4GB RAM, €9/month)
5. Add SSH key
6. Name: `fluxhero-production`
7. Create server

**Via hcloud CLI**:

```bash
# Install hcloud CLI
brew install hcloud  # macOS

# Authenticate
hcloud context create fluxhero

# Create server
hcloud server create \
  --type cpx21 \
  --image ubuntu-22.04 \
  --location nbg1 \
  --name fluxhero-production \
  --ssh-key YOUR_SSH_KEY_ID
```

### Step 2: Configure Firewall

```bash
# Create firewall
hcloud firewall create --name fluxhero-firewall

# Add rules
hcloud firewall add-rule fluxhero-firewall \
  --direction in --protocol tcp --port 22 --source-ips YOUR_IP/32

hcloud firewall add-rule fluxhero-firewall \
  --direction in --protocol tcp --port 80 --source-ips 0.0.0.0/0

hcloud firewall add-rule fluxhero-firewall \
  --direction in --protocol tcp --port 443 --source-ips 0.0.0.0/0

# Apply to server
hcloud firewall apply-to-resource fluxhero-firewall \
  --type server --server fluxhero-production
```

### Step 3: Connect

```bash
ssh root@YOUR_SERVER_IP
```

**Continue to [Common Setup Steps](#common-setup-steps)**

---

## Common Setup Steps

After connecting to your VPS (any provider), follow these steps:

### Step 1: Update System and Install Dependencies

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
  python3.11 \
  python3.11-venv \
  python3-pip \
  git \
  nginx \
  certbot \
  python3-certbot-nginx \
  build-essential \
  curl \
  htop \
  ufw

# Install Node.js 20.x for frontend
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installations
python3.11 --version  # Should show Python 3.11.x
node --version        # Should show v20.x.x
npm --version         # Should show 10.x.x
```

### Step 2: Create Deployment User

```bash
# Create non-root user for security
sudo adduser fluxhero
sudo usermod -aG sudo fluxhero

# Copy SSH keys
sudo mkdir -p /home/fluxhero/.ssh
sudo cp ~/.ssh/authorized_keys /home/fluxhero/.ssh/
sudo chown -R fluxhero:fluxhero /home/fluxhero/.ssh
sudo chmod 700 /home/fluxhero/.ssh
sudo chmod 600 /home/fluxhero/.ssh/authorized_keys

# Switch to fluxhero user
sudo su - fluxhero
```

### Step 3: Clone and Setup FluxHero

```bash
# Clone repository
cd ~
git clone https://github.com/YOUR_USERNAME/fluxhero.git
# Or if using SSH
# git clone git@github.com:YOUR_USERNAME/fluxhero.git

cd fluxhero

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup backend
cd backend
uv venv .venv
source .venv/bin/activate
uv sync

# Setup frontend
cd ../frontend
npm install

# Return to root
cd ~
```

### Step 4: Configure Environment Variables

```bash
# Create backend .env file
cat > ~/fluxhero/backend/.env << 'EOF'
# API Configuration
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use https://api.alpaca.markets for live

# Database
DB_PATH=data/fluxhero.db

# Risk Management
MAX_DAILY_LOSS_PERCENT=3.0
MAX_POSITION_SIZE_PERCENT=20.0
MAX_TOTAL_EXPOSURE_PERCENT=50.0

# System
LOG_LEVEL=INFO
TIMEZONE=America/New_York

# API Server
API_HOST=0.0.0.0
API_PORT=8000
EOF

# Create frontend .env.local file
cat > ~/fluxhero/frontend/.env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
EOF

# Secure the files
chmod 600 ~/fluxhero/backend/.env
chmod 600 ~/fluxhero/frontend/.env.local
```

### Step 5: Initialize Database and Data Directories

```bash
cd ~/fluxhero/backend

# Create data directories
mkdir -p data/market_data
mkdir -p data/logs
mkdir -p data/backups

# Initialize database
source venv/bin/activate
python -c "from storage.sqlite_store import SQLiteStore; store = SQLiteStore('data/fluxhero.db'); print('Database initialized')"
```

### Step 6: Build Frontend

```bash
cd ~/fluxhero/frontend
npm run build
```

---

## Process Management

Use **systemd** to manage FluxHero as a system service.

### Backend Service

Create systemd service file:

```bash
sudo nano /etc/systemd/system/fluxhero-backend.service
```

Add content:

```ini
[Unit]
Description=FluxHero Backend API
After=network.target

[Service]
Type=simple
User=fluxhero
WorkingDirectory=/home/fluxhero/fluxhero/backend
Environment="PATH=/home/fluxhero/fluxhero/backend/venv/bin"
ExecStart=/home/fluxhero/fluxhero/backend/venv/bin/python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/home/fluxhero/fluxhero/backend/data/logs/backend.log
StandardError=append:/home/fluxhero/fluxhero/backend/data/logs/backend-error.log

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### Frontend Service

```bash
sudo nano /etc/systemd/system/fluxhero-frontend.service
```

Add content:

```ini
[Unit]
Description=FluxHero Frontend (Next.js)
After=network.target fluxhero-backend.service

[Service]
Type=simple
User=fluxhero
WorkingDirectory=/home/fluxhero/fluxhero/frontend
Environment="PATH=/usr/bin:/usr/local/bin"
Environment="NODE_ENV=production"
ExecStart=/usr/bin/npm run start
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/home/fluxhero/fluxhero/backend/data/logs/frontend.log
StandardError=append:/home/fluxhero/fluxhero/backend/data/logs/frontend-error.log

[Install]
WantedBy=multi-user.target
```

### Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable fluxhero-backend
sudo systemctl enable fluxhero-frontend

# Start services
sudo systemctl start fluxhero-backend
sudo systemctl start fluxhero-frontend

# Check status
sudo systemctl status fluxhero-backend
sudo systemctl status fluxhero-frontend

# View logs
sudo journalctl -u fluxhero-backend -f
sudo journalctl -u fluxhero-frontend -f
```

---

## Security Hardening

### Step 1: Configure UFW Firewall

```bash
# Enable UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (change port if using non-standard)
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

### Step 2: Secure SSH

```bash
sudo nano /etc/ssh/sshd_config
```

Update these settings:

```
# Disable root login
PermitRootLogin no

# Disable password authentication (SSH keys only)
PasswordAuthentication no

# Disable empty passwords
PermitEmptyPasswords no

# Change default port (optional but recommended)
Port 2222  # Use custom port

# Allow only specific user
AllowUsers fluxhero
```

Restart SSH:

```bash
sudo systemctl restart sshd
```

**Important**: If you changed the SSH port, update UFW and reconnect using the new port:

```bash
sudo ufw allow 2222/tcp
ssh -p 2222 fluxhero@YOUR_SERVER_IP
```

### Step 3: Setup Fail2Ban

```bash
# Install fail2ban
sudo apt install -y fail2ban

# Create local config
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local
```

Update settings:

```ini
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 5

[sshd]
enabled = true
port = 2222  # Match your SSH port
```

Start fail2ban:

```bash
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### Step 4: Setup Automatic Security Updates

```bash
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

---

## SSL/TLS Configuration

### Option A: Using Certbot (Let's Encrypt) - Free

**Prerequisites**: You need a domain name pointing to your server's IP.

```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Certbot will automatically:
# 1. Obtain certificate from Let's Encrypt
# 2. Configure Nginx with SSL
# 3. Setup auto-renewal
```

### Option B: Manual Nginx Configuration with SSL

If you already have SSL certificates:

```bash
sudo nano /etc/nginx/sites-available/fluxhero
```

Add configuration:

```nginx
# HTTP - Redirect to HTTPS
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Frontend (Next.js)
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://localhost:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
```

Enable and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/fluxhero /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

---

## Monitoring & Alerting

### Step 1: Setup Log Rotation

```bash
sudo nano /etc/logrotate.d/fluxhero
```

Add:

```
/home/fluxhero/fluxhero/backend/data/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 fluxhero fluxhero
}
```

### Step 2: Setup System Monitoring Script

```bash
nano ~/fluxhero/scripts/monitor.sh
```

Add:

```bash
#!/bin/bash

# Check if backend is running
if ! systemctl is-active --quiet fluxhero-backend; then
    echo "Backend is down! Restarting..."
    systemctl restart fluxhero-backend
    # Send alert email (configure mail command first)
    # echo "FluxHero backend restarted" | mail -s "FluxHero Alert" your-email@example.com
fi

# Check if frontend is running
if ! systemctl is-active --quiet fluxhero-frontend; then
    echo "Frontend is down! Restarting..."
    systemctl restart fluxhero-frontend
fi

# Check disk space
DISK_USAGE=$(df -h /home/fluxhero | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "Disk usage critical: ${DISK_USAGE}%"
    # Send alert
fi

# Check if database is accessible
if ! sqlite3 /home/fluxhero/fluxhero/backend/data/fluxhero.db "SELECT 1;" > /dev/null 2>&1; then
    echo "Database check failed!"
    # Send alert
fi
```

Make executable and add to crontab:

```bash
chmod +x ~/fluxhero/scripts/monitor.sh

# Add to crontab (run every 5 minutes)
crontab -e
```

Add line:

```
*/5 * * * * /home/fluxhero/fluxhero/scripts/monitor.sh >> /home/fluxhero/fluxhero/backend/data/logs/monitor.log 2>&1
```

### Step 3: Setup Email Alerts (Optional)

```bash
# Install mailutils
sudo apt install -y mailutils

# Configure (follow prompts)
sudo dpkg-reconfigure postfix
```

---

## Backup & Disaster Recovery

### Step 1: Setup Automated Backups

```bash
nano ~/fluxhero/scripts/backup.sh
```

Add:

```bash
#!/bin/bash

BACKUP_DIR="/home/fluxhero/fluxhero/backend/data/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_FILE="/home/fluxhero/fluxhero/backend/data/fluxhero.db"

# Create backup directory if not exists
mkdir -p "$BACKUP_DIR"

# Backup database
sqlite3 "$DB_FILE" ".backup '$BACKUP_DIR/fluxhero_$DATE.db'"

# Backup parquet files
tar -czf "$BACKUP_DIR/market_data_$DATE.tar.gz" /home/fluxhero/fluxhero/backend/data/market_data/

# Backup environment files
cp /home/fluxhero/fluxhero/backend/.env "$BACKUP_DIR/env_$DATE.backup"

# Remove backups older than 30 days
find "$BACKUP_DIR" -name "*.db" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

Make executable and schedule:

```bash
chmod +x ~/fluxhero/scripts/backup.sh

# Add to crontab (daily at 2 AM)
crontab -e
```

Add:

```
0 2 * * * /home/fluxhero/fluxhero/scripts/backup.sh >> /home/fluxhero/fluxhero/backend/data/logs/backup.log 2>&1
```

### Step 2: Offsite Backup to S3 (Optional)

```bash
# Install AWS CLI
sudo apt install -y awscli

# Configure AWS credentials
aws configure

# Modify backup script to include S3 upload
# Add to end of backup.sh:
# aws s3 cp "$BACKUP_DIR/fluxhero_$DATE.db" s3://your-bucket-name/backups/
```

### Step 3: Disaster Recovery Procedure

**To restore from backup**:

```bash
# Stop services
sudo systemctl stop fluxhero-backend
sudo systemctl stop fluxhero-frontend

# Restore database
cp /home/fluxhero/fluxhero/backend/data/backups/fluxhero_YYYYMMDD_HHMMSS.db \
   /home/fluxhero/fluxhero/backend/data/fluxhero.db

# Restore market data
tar -xzf /home/fluxhero/fluxhero/backend/data/backups/market_data_YYYYMMDD_HHMMSS.tar.gz \
    -C /home/fluxhero/fluxhero/backend/data/

# Restart services
sudo systemctl start fluxhero-backend
sudo systemctl start fluxhero-frontend
```

---

## Cost Optimization

### 1. Use Reserved Instances (AWS)

- **Savings**: ~30-50% vs on-demand
- **Commitment**: 1-3 years
- **Best for**: Long-term production use

### 2. Use Spot Instances (AWS) - **NOT RECOMMENDED FOR PRODUCTION**

- **Savings**: ~70-90% vs on-demand
- **Risk**: Can be terminated anytime
- **Use case**: Backtesting only, not live trading

### 3. Snapshot Scheduling

- Only keep snapshots from last 7 days
- Use lifecycle policies to auto-delete old backups

### 4. Optimize Storage

```bash
# Compress old logs
find /home/fluxhero/fluxhero/backend/data/logs -name "*.log" -mtime +7 -exec gzip {} \;

# Remove old parquet cache
find /home/fluxhero/fluxhero/backend/data/market_data -name "*.parquet" -mtime +90 -delete
```

### 5. Monitor Data Transfer

- Keep backend and frontend on same instance (avoid cross-region transfer fees)
- Use CloudFront/CDN only if serving globally

---

## Troubleshooting

### Issue: Backend won't start

```bash
# Check logs
sudo journalctl -u fluxhero-backend -n 50

# Common fixes:
# 1. Check if port 8000 is already in use
sudo lsof -i :8000

# 2. Check environment variables
cat ~/fluxhero/backend/.env

# 3. Check database permissions
ls -la ~/fluxhero/backend/data/fluxhero.db
```

### Issue: Frontend build fails

```bash
# Check Node version
node --version  # Should be 20.x

# Clear cache and rebuild
cd ~/fluxhero/frontend
rm -rf .next node_modules
npm install
npm run build
```

### Issue: SSL certificate renewal fails

```bash
# Test renewal
sudo certbot renew --dry-run

# Force renewal
sudo certbot renew --force-renewal

# Check Nginx config
sudo nginx -t
```

### Issue: High CPU usage

```bash
# Check process usage
htop

# Check if Numba compilation is happening
# (Normal on first run, should cache afterward)

# Check for infinite loops in logs
tail -f ~/fluxhero/backend/data/logs/backend.log
```

### Issue: Database locked

```bash
# Check for zombie processes
ps aux | grep python

# Kill if needed
sudo killall -9 python

# Restart backend
sudo systemctl restart fluxhero-backend
```

---

## Next Steps

After deployment:

1. **Test the system**: Access via `https://yourdomain.com` and verify all features work
2. **Setup monitoring**: Configure email/SMS alerts for critical failures
3. **Run paper trading**: Test with paper account for 1-2 weeks before live trading
4. **Review logs daily**: Check for errors, warnings, or unusual patterns
5. **Setup scheduled maintenance**: Monthly updates, quarterly reviews
6. **Document your changes**: Keep notes on custom configurations

### Related Documentation

- [User Guide](USER_GUIDE.md) - How to use FluxHero
- [Risk Management](RISK_MANAGEMENT.md) - Understanding risk rules
- [Maintenance Guide](MAINTENANCE_GUIDE.md) - Ongoing maintenance tasks
- [API Documentation](API_DOCUMENTATION.md) - API reference

---

**Need Help?**

- Check logs: `sudo journalctl -u fluxhero-backend -f`
- Review error messages in `/home/fluxhero/fluxhero/backend/data/logs/`
- Consult the troubleshooting section above
- Review system requirements in README.md

---

**Last Updated**: 2026-01-21
**Version**: 1.0.0
