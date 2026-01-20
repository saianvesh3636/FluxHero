"""
Position Sizer Module

Implements risk-based position sizing with account-level limits and safety controls.

Requirements implemented (from FLUXHERO_REQUIREMENTS.md):
- R10.3.1: 1% Risk Rule calculation
- R10.3.2: Max position size limits (20% per position, 50% total deployment)
- R10.3.3: Round down to nearest whole share
- R10.4.1: Track daily P&L since market open
- R10.4.2: Kill-switch at 3% daily loss
- R10.4.3: Manual override for re-enabling trading

Author: FluxHero Team
Date: 2026-01-20
"""

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Optional


class PositionSizeResult(IntEnum):
    """Result codes for position sizing calculations."""
    SUCCESS = 0
    INSUFFICIENT_CAPITAL = 1
    EXCEEDS_POSITION_LIMIT = 2
    EXCEEDS_TOTAL_DEPLOYMENT = 3
    KILL_SWITCH_ACTIVE = 4
    INVALID_RISK = 5


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    shares: int  # Number of shares to trade (0 if rejected)
    risk_amount: float  # Dollar risk per trade
    position_value: float  # Total position value ($)
    result_code: PositionSizeResult  # Result status
    reason: str  # Human-readable explanation


@dataclass
class AccountState:
    """Current account state for position sizing."""
    balance: float  # Total account value ($)
    cash: float  # Available cash ($)
    deployed_value: float  # Total value of open positions ($)
    daily_pnl: float  # P&L since market open ($)
    num_positions: int  # Number of open positions
    session_start_balance: float  # Balance at session start (for daily P&L tracking)


class PositionSizer:
    """
    Risk-based position sizer with account-level limits and safety controls.

    Features:
    - 1% risk rule: Risk 1% of account per trade
    - Max position size: 20% of account per position
    - Max total deployment: 50% of account
    - Kill-switch: Stop trading if daily loss > 3%
    - Whole share rounding

    Example:
        >>> sizer = PositionSizer(risk_pct=0.01, max_position_pct=0.20, max_deployment_pct=0.50)
        >>> account = AccountState(balance=10000, cash=8000, deployed_value=2000,
        ...                        daily_pnl=-100, num_positions=1, session_start_balance=10100)
        >>> result = sizer.calculate_position_size(
        ...     account=account,
        ...     entry_price=100.0,
        ...     stop_loss_price=98.0,
        ...     strategy='trend'
        ... )
        >>> print(f"Shares: {result.shares}, Risk: ${result.risk_amount:.2f}")
        Shares: 50, Risk: $100.00
    """

    def __init__(
        self,
        risk_pct: float = 0.01,  # R10.3.1: 1% risk per trade
        max_position_pct: float = 0.20,  # R10.3.2: Max 20% per position
        max_deployment_pct: float = 0.50,  # R10.3.2: Max 50% total deployed
        kill_switch_pct: float = 0.03,  # R10.4.2: Kill-switch at 3% daily loss
        kill_switch_enabled: bool = True,  # R10.4.3: Can be disabled manually
    ):
        """
        Initialize position sizer with risk parameters.

        Args:
            risk_pct: Percentage of account to risk per trade (default: 0.01 = 1%)
            max_position_pct: Max position size as % of account (default: 0.20 = 20%)
            max_deployment_pct: Max total deployed capital as % (default: 0.50 = 50%)
            kill_switch_pct: Daily loss threshold for kill-switch (default: 0.03 = 3%)
            kill_switch_enabled: Whether kill-switch is active (default: True)
        """
        self.risk_pct = risk_pct
        self.max_position_pct = max_position_pct
        self.max_deployment_pct = max_deployment_pct
        self.kill_switch_pct = kill_switch_pct
        self.kill_switch_enabled = kill_switch_enabled
        self.kill_switch_triggered = False  # Track if kill-switch has fired
        self.kill_switch_trigger_time: Optional[datetime] = None

    def calculate_position_size(
        self,
        account: AccountState,
        entry_price: float,
        stop_loss_price: float,
        strategy: str = 'trend',
    ) -> PositionSize:
        """
        Calculate position size based on 1% risk rule with account-level limits.

        Requirements:
        - R10.3.1: shares = (account_balance × risk_pct) / (entry_price - stop_loss_price)
        - R10.3.2: Never exceed max_position_pct (20%) or max_deployment_pct (50%)
        - R10.3.3: Round down to nearest whole share
        - R10.4.2: Kill-switch if daily loss > kill_switch_pct (3%)

        Args:
            account: Current account state
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            strategy: Strategy name ('trend' or 'mean_reversion')

        Returns:
            PositionSize object with shares and metadata
        """
        # R10.4.2: Check kill-switch (daily loss > 3%)
        if self.kill_switch_enabled and self._check_kill_switch(account):
            return PositionSize(
                shares=0,
                risk_amount=0.0,
                position_value=0.0,
                result_code=PositionSizeResult.KILL_SWITCH_ACTIVE,
                reason=f"Kill-switch active: Daily loss {account.daily_pnl:.2f} exceeds -{self.kill_switch_pct*100:.1f}%"
            )

        # Validate inputs
        if entry_price <= 0 or stop_loss_price <= 0:
            return PositionSize(
                shares=0,
                risk_amount=0.0,
                position_value=0.0,
                result_code=PositionSizeResult.INVALID_RISK,
                reason="Invalid entry or stop loss price (must be > 0)"
            )

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share == 0:
            return PositionSize(
                shares=0,
                risk_amount=0.0,
                position_value=0.0,
                result_code=PositionSizeResult.INVALID_RISK,
                reason="Invalid risk: entry price equals stop loss price"
            )

        # R10.3.1: 1% Risk Rule
        # shares = (account_balance × risk_pct) / (entry_price - stop_loss_price)
        risk_amount = account.balance * self.risk_pct
        shares_by_risk = risk_amount / risk_per_share

        # R10.3.2: Apply max position size limit (20% of account)
        max_position_value = account.balance * self.max_position_pct
        shares_by_position_limit = max_position_value / entry_price

        # Take minimum of risk-based and position-limit-based sizing
        shares = min(shares_by_risk, shares_by_position_limit)

        # R10.3.3: Round down to nearest whole share
        shares = int(shares)

        # Check if we have enough shares (minimum 1)
        if shares < 1:
            return PositionSize(
                shares=0,
                risk_amount=0.0,
                position_value=0.0,
                result_code=PositionSizeResult.INSUFFICIENT_CAPITAL,
                reason=f"Position too small: Risk-based calculation yielded {shares_by_risk:.2f} shares"
            )

        # Calculate position value
        position_value = shares * entry_price

        # Check if we have enough cash
        if position_value > account.cash:
            # Reduce shares to fit available cash
            shares = int(account.cash / entry_price)
            position_value = shares * entry_price

            if shares < 1:
                return PositionSize(
                    shares=0,
                    risk_amount=0.0,
                    position_value=0.0,
                    result_code=PositionSizeResult.INSUFFICIENT_CAPITAL,
                    reason=f"Insufficient cash: Need ${position_value:.2f}, have ${account.cash:.2f}"
                )

        # R10.3.2: Check total deployment limit (50% of account)
        new_deployed_value = account.deployed_value + position_value
        max_deployed_value = account.balance * self.max_deployment_pct

        if new_deployed_value > max_deployed_value:
            # Calculate how many shares we can add without exceeding limit
            available_deployment = max_deployed_value - account.deployed_value

            if available_deployment <= 0:
                return PositionSize(
                    shares=0,
                    risk_amount=0.0,
                    position_value=0.0,
                    result_code=PositionSizeResult.EXCEEDS_TOTAL_DEPLOYMENT,
                    reason=f"Total deployment limit reached: {account.deployed_value:.2f}/{max_deployed_value:.2f}"
                )

            # Reduce shares to fit within deployment limit
            shares = int(available_deployment / entry_price)
            position_value = shares * entry_price

            if shares < 1:
                return PositionSize(
                    shares=0,
                    risk_amount=0.0,
                    position_value=0.0,
                    result_code=PositionSizeResult.EXCEEDS_TOTAL_DEPLOYMENT,
                    reason=f"Position would exceed deployment limit: Current {account.deployed_value:.2f}, Max {max_deployed_value:.2f}"
                )

        # Calculate actual risk amount with final share count
        actual_risk_amount = shares * risk_per_share

        return PositionSize(
            shares=shares,
            risk_amount=actual_risk_amount,
            position_value=position_value,
            result_code=PositionSizeResult.SUCCESS,
            reason=f"Success: {shares} shares @ ${entry_price:.2f} = ${position_value:.2f}, Risk: ${actual_risk_amount:.2f}"
        )

    def _check_kill_switch(self, account: AccountState) -> bool:
        """
        Check if kill-switch should be triggered based on daily P&L.

        Requirements:
        - R10.4.1: Track daily P&L since market open
        - R10.4.2: Trigger if daily loss exceeds 3% of account

        Args:
            account: Current account state

        Returns:
            True if kill-switch should trigger, False otherwise
        """
        # Calculate daily loss percentage
        daily_loss_pct = account.daily_pnl / account.session_start_balance

        # Trigger if daily loss exceeds threshold (negative P&L)
        if daily_loss_pct <= -self.kill_switch_pct:
            if not self.kill_switch_triggered:
                self.kill_switch_triggered = True
                self.kill_switch_trigger_time = datetime.now()
            return True

        return False

    def reset_kill_switch(self) -> None:
        """
        Reset kill-switch (manual override).

        Requirements:
        - R10.4.3: Manual override to re-enable trading

        Should be called at start of new trading session or by explicit user action.
        """
        self.kill_switch_triggered = False
        self.kill_switch_trigger_time = None

    def enable_kill_switch(self) -> None:
        """Enable kill-switch protection."""
        self.kill_switch_enabled = True

    def disable_kill_switch(self) -> None:
        """
        Disable kill-switch protection (R10.4.3: manual override).

        WARNING: This removes safety protection. Use with caution.
        """
        self.kill_switch_enabled = False
        self.kill_switch_triggered = False

    def get_max_shares(
        self,
        account: AccountState,
        entry_price: float,
    ) -> int:
        """
        Calculate maximum shares allowed by position and deployment limits.

        Args:
            account: Current account state
            entry_price: Entry price for the trade

        Returns:
            Maximum number of shares allowed
        """
        # Check position limit (20% of account)
        max_position_value = account.balance * self.max_position_pct
        shares_by_position = int(max_position_value / entry_price)

        # Check deployment limit (50% of account)
        available_deployment = (account.balance * self.max_deployment_pct) - account.deployed_value
        shares_by_deployment = int(available_deployment / entry_price) if available_deployment > 0 else 0

        # Check cash limit
        shares_by_cash = int(account.cash / entry_price)

        # Return minimum of all limits
        return max(0, min(shares_by_position, shares_by_deployment, shares_by_cash))

    def get_risk_metrics(self, account: AccountState) -> dict:
        """
        Get current risk metrics for monitoring.

        Returns:
            Dictionary with risk metrics:
            - deployment_pct: Current deployment as % of account
            - deployment_used: Dollar value deployed
            - deployment_available: Dollar value available
            - daily_pnl: Daily P&L
            - daily_pnl_pct: Daily P&L as % of session start
            - kill_switch_triggered: Whether kill-switch is active
            - kill_switch_distance: Distance to kill-switch threshold
        """
        deployment_pct = account.deployed_value / account.balance if account.balance > 0 else 0
        max_deployment = account.balance * self.max_deployment_pct
        deployment_available = max(0, max_deployment - account.deployed_value)

        daily_pnl_pct = account.daily_pnl / account.session_start_balance if account.session_start_balance > 0 else 0
        kill_switch_distance = daily_pnl_pct - (-self.kill_switch_pct)  # Distance to threshold (negative = safe)

        return {
            'deployment_pct': deployment_pct,
            'deployment_used': account.deployed_value,
            'deployment_available': deployment_available,
            'daily_pnl': account.daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'kill_switch_triggered': self.kill_switch_triggered,
            'kill_switch_distance': kill_switch_distance,
        }
