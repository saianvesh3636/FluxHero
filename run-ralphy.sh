#!/bin/bash
# FluxHero Ralphy Execution Script
# This script runs Ralphy with the correct settings for FluxHero development

# Configuration
RALPHY_PATH="/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh"
TASK_FILE="FLUXHERO_TASKS.md"
PROJECT_DIR="/Users/anvesh/Developer/QuantTrading/project"

# Execution settings (based on user preferences)
# - Sequential execution (no --parallel flag)
# - NO branch per task (continue on main branch to avoid conflicts)
# - No auto PRs (no --create-pr flag)
# - Using Claude Code (default AI engine)

cd "$PROJECT_DIR" || exit 1

# Ensure we're on main branch
git checkout main 2>/dev/null || true

echo "🚀 Starting FluxHero implementation with Ralphy"
echo "📋 Task file: $TASK_FILE"
echo "🌳 Branch strategy: Single main branch (simpler workflow)"
echo "⚙️  Execution mode: Sequential (one task at a time)"
echo ""

# Run Ralphy WITHOUT branch-per-task to avoid branch confusion
"$RALPHY_PATH" \
  --prd "$TASK_FILE" \
  --base-branch main

echo ""
echo "✅ Ralphy execution completed!"
echo "📊 Check .ralphy/progress.txt for detailed logs"
