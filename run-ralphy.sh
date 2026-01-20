#!/bin/bash
# FluxHero Ralphy Execution Script
# This script runs Ralphy with the correct settings for FluxHero development

# Configuration
RALPHY_PATH="/Users/anvesh/Developer/QuantTrading/ralphy/ralphy.sh"
TASK_FILE="FLUXHERO_TASKS.md"
PROJECT_DIR="/Users/anvesh/Developer/QuantTrading/project"

# Execution settings (based on user preferences)
# - Sequential execution (no --parallel flag)
# - Branch per task (--branch-per-task)
# - No auto PRs (no --create-pr flag)
# - Using Claude Code (default AI engine)

cd "$PROJECT_DIR" || exit 1

echo "🚀 Starting FluxHero implementation with Ralphy"
echo "📋 Task file: $TASK_FILE"
echo "🌳 Branch strategy: Separate branch per task"
echo "⚙️  Execution mode: Sequential (one task at a time)"
echo ""

# Run Ralphy with FluxHero configuration
"$RALPHY_PATH" \
  --prd "$TASK_FILE" \
  --branch-per-task \
  --base-branch main

echo ""
echo "✅ Ralphy execution completed!"
echo "📊 Check .ralphy/progress.txt for detailed logs"
