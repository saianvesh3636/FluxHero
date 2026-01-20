# FluxHero - Implementation

This folder will contain the actual implementation of the FluxHero trading system.

**FluxHero**: Where flux (constant adaptation) meets hero (mastery).

## Status

**NOT STARTED**

Currently in planning/ideation phase. See `/workspace/` for all planning materials.

## When Ready to Start

This folder will contain:

```
project/
├── data/                    # Data pipeline
│   ├── fetchers/           # Data source connectors
│   ├── processors/         # Data cleaning, validation
│   └── storage/            # Data persistence
├── strategies/              # Trading strategies
│   ├── base.py             # Base strategy interface
│   └── [strategy_name]/    # Individual strategies
├── backtesting/            # Backtesting engine
├── execution/              # Live trading execution (future)
├── risk/                   # Risk management
├── utils/                  # Shared utilities
├── tests/                  # Test suites
├── config/                 # Configuration
├── notebooks/              # Jupyter notebooks for exploration
└── requirements.txt        # Python dependencies
```

## Before Starting Implementation

Make sure you have:
1. Reviewed rules in `/workspace/ai-guidelines/`
2. Read research materials in `/workspace/research/`
3. Decided on first strategy to implement
4. Understood the "brick by brick" approach

## Implementation Rules

When implementing, follow:
- **Grounding rules**: `/workspace/ai-guidelines/grounding-rules.md`
- **Coding standards**: `/workspace/ai-guidelines/coding-standards.md`
- **Claude behavior**: `/workspace/ai-guidelines/claude-behavior-rules.md`

## Getting Started (When Ready)

1. Set up Python environment
2. Create `requirements.txt`
3. Implement first "brick" (likely data fetching for SPY)
4. Build incrementally from there

## Current Phase

**Phase**: Ideation and Planning
**Next Phase**: Implementation (when you decide to start)
**Location**: All planning work is in `/workspace/`
