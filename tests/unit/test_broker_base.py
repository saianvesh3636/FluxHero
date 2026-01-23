"""
Unit tests for broker base class (abstract BrokerInterface).

Tests cover:
- Abstract interface enforcement
- BrokerInterface cannot be instantiated
- BrokerHealth dataclass functionality
- Connection lifecycle methods (connect, disconnect, health_check)
- Import from both broker_base and broker_interface (backward compatibility)
"""

import pytest

from backend.execution.broker_base import (
    Account,
    BrokerHealth,
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from backend.execution.broker_interface import PaperBroker


# Test BrokerInterface abstract class enforcement
class TestBrokerInterfaceAbstract:
    """Tests for abstract BrokerInterface enforcement."""

    def test_broker_interface_cannot_be_instantiated(self):
        """Test that BrokerInterface cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BrokerInterface()

    def test_broker_interface_requires_all_methods(self):
        """Test that partial implementation raises TypeError."""

        class PartialBroker(BrokerInterface):
            async def connect(self) -> bool:
                return True

            async def disconnect(self) -> None:
                pass

            # Missing: health_check, get_account, get_positions, place_order,
            #          cancel_order, get_order_status

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PartialBroker()

    def test_complete_implementation_can_be_instantiated(self):
        """Test that complete implementation can be instantiated."""

        class CompleteBroker(BrokerInterface):
            async def connect(self) -> bool:
                return True

            async def disconnect(self) -> None:
                pass

            async def health_check(self) -> BrokerHealth:
                return BrokerHealth(is_connected=True, is_authenticated=True)

            async def get_account(self) -> Account:
                return Account(
                    account_id="TEST",
                    balance=100000.0,
                    buying_power=100000.0,
                    equity=100000.0,
                    cash=100000.0,
                )

            async def get_positions(self) -> list[Position]:
                return []

            async def place_order(
                self,
                symbol: str,
                qty: int,
                side: OrderSide,
                order_type: OrderType,
                limit_price: float | None = None,
                stop_price: float | None = None,
            ) -> Order:
                return Order(
                    order_id="TEST_001",
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price,
                )

            async def cancel_order(self, order_id: str) -> bool:
                return True

            async def get_order_status(self, order_id: str) -> Order | None:
                return None

        broker = CompleteBroker()
        assert broker is not None


# Test BrokerHealth dataclass
class TestBrokerHealth:
    """Tests for BrokerHealth dataclass."""

    def test_broker_health_creation(self):
        """Test BrokerHealth can be created with required fields."""
        health = BrokerHealth(is_connected=True, is_authenticated=True)

        assert health.is_connected is True
        assert health.is_authenticated is True
        assert health.latency_ms is None
        assert health.last_heartbeat is None
        assert health.error_message is None

    def test_broker_health_with_all_fields(self):
        """Test BrokerHealth with all optional fields."""
        health = BrokerHealth(
            is_connected=True,
            is_authenticated=True,
            latency_ms=15.5,
            last_heartbeat=1234567890.0,
            error_message=None,
        )

        assert health.latency_ms == 15.5
        assert health.last_heartbeat == 1234567890.0

    def test_broker_health_unhealthy_state(self):
        """Test BrokerHealth for unhealthy connection."""
        health = BrokerHealth(
            is_connected=False,
            is_authenticated=False,
            error_message="Connection refused",
        )

        assert health.is_connected is False
        assert health.is_authenticated is False
        assert health.error_message == "Connection refused"

    def test_is_healthy_property_true(self):
        """Test is_healthy returns True when connected and authenticated."""
        health = BrokerHealth(is_connected=True, is_authenticated=True)

        assert health.is_healthy is True

    def test_is_healthy_property_false_not_connected(self):
        """Test is_healthy returns False when not connected."""
        health = BrokerHealth(is_connected=False, is_authenticated=True)

        assert health.is_healthy is False

    def test_is_healthy_property_false_not_authenticated(self):
        """Test is_healthy returns False when not authenticated."""
        health = BrokerHealth(is_connected=True, is_authenticated=False)

        assert health.is_healthy is False

    def test_is_healthy_property_false_both(self):
        """Test is_healthy returns False when both conditions fail."""
        health = BrokerHealth(is_connected=False, is_authenticated=False)

        assert health.is_healthy is False


# Test PaperBroker connection lifecycle methods
class TestPaperBrokerConnectionLifecycle:
    """Tests for PaperBroker connection lifecycle methods."""

    @pytest.mark.asyncio
    async def test_connect_returns_true(self):
        """Test PaperBroker.connect() returns True."""
        broker = PaperBroker()

        result = await broker.connect()

        assert result is True

    @pytest.mark.asyncio
    async def test_connect_sets_connected_flag(self):
        """Test connect() sets internal connected flag."""
        broker = PaperBroker()
        assert broker._connected is False

        await broker.connect()

        assert broker._connected is True

    @pytest.mark.asyncio
    async def test_connect_sets_last_heartbeat(self):
        """Test connect() sets last heartbeat timestamp."""
        broker = PaperBroker()
        assert broker._last_heartbeat is None

        await broker.connect()

        assert broker._last_heartbeat is not None
        assert broker._last_heartbeat > 0

    @pytest.mark.asyncio
    async def test_disconnect_clears_connected_flag(self):
        """Test disconnect() clears connected flag."""
        broker = PaperBroker()
        await broker.connect()
        assert broker._connected is True

        await broker.disconnect()

        assert broker._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_clears_last_heartbeat(self):
        """Test disconnect() clears last heartbeat."""
        broker = PaperBroker()
        await broker.connect()
        assert broker._last_heartbeat is not None

        await broker.disconnect()

        assert broker._last_heartbeat is None

    @pytest.mark.asyncio
    async def test_disconnect_safe_when_not_connected(self):
        """Test disconnect() is safe to call when not connected."""
        broker = PaperBroker()

        # Should not raise
        await broker.disconnect()

        assert broker._connected is False

    @pytest.mark.asyncio
    async def test_health_check_when_connected(self):
        """Test health_check() returns healthy status when connected."""
        broker = PaperBroker()
        await broker.connect()

        health = await broker.health_check()

        assert health.is_connected is True
        assert health.is_authenticated is True
        assert health.is_healthy is True
        assert health.latency_ms == 0.0  # Paper broker has no latency
        assert health.last_heartbeat is not None
        assert health.error_message is None

    @pytest.mark.asyncio
    async def test_health_check_when_not_connected(self):
        """Test health_check() returns unhealthy status when not connected."""
        broker = PaperBroker()

        health = await broker.health_check()

        assert health.is_connected is False
        assert health.is_authenticated is False
        assert health.is_healthy is False
        assert health.latency_ms is None
        assert health.error_message == "Not connected"

    @pytest.mark.asyncio
    async def test_health_check_updates_heartbeat(self):
        """Test health_check() updates last heartbeat timestamp."""
        import time

        broker = PaperBroker()
        await broker.connect()

        initial_heartbeat = broker._last_heartbeat
        time.sleep(0.01)  # Small delay

        await broker.health_check()

        assert broker._last_heartbeat >= initial_heartbeat

    @pytest.mark.asyncio
    async def test_health_check_after_disconnect(self):
        """Test health_check() after disconnect shows unhealthy."""
        broker = PaperBroker()
        await broker.connect()

        health1 = await broker.health_check()
        assert health1.is_healthy is True

        await broker.disconnect()

        health2 = await broker.health_check()
        assert health2.is_healthy is False

    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect(self):
        """Test can reconnect after disconnect."""
        broker = PaperBroker()
        await broker.connect()
        await broker.disconnect()

        result = await broker.connect()

        assert result is True
        assert broker._connected is True

        health = await broker.health_check()
        assert health.is_healthy is True


# Test backward compatibility imports
class TestBackwardCompatibility:
    """Tests for backward compatibility with broker_interface imports."""

    def test_import_from_broker_interface(self):
        """Test all types can still be imported from broker_interface."""
        from backend.execution.broker_interface import (
            Account,
            BrokerHealth,
            BrokerInterface,
            Order,
            OrderSide,
            OrderStatus,
            OrderType,
            PaperBroker,
            Position,
        )

        # Verify imports work
        assert Account is not None
        assert BrokerHealth is not None
        assert BrokerInterface is not None
        assert Order is not None
        assert OrderSide is not None
        assert OrderStatus is not None
        assert OrderType is not None
        assert PaperBroker is not None
        assert Position is not None

    def test_import_from_broker_base(self):
        """Test all base types can be imported from broker_base."""
        from backend.execution.broker_base import (
            Account,
            BrokerHealth,
            BrokerInterface,
            Order,
            OrderSide,
            OrderStatus,
            OrderType,
            Position,
        )

        # Verify imports work
        assert Account is not None
        assert BrokerHealth is not None
        assert BrokerInterface is not None
        assert Order is not None
        assert OrderSide is not None
        assert OrderStatus is not None
        assert OrderType is not None
        assert Position is not None

    def test_types_are_same_objects(self):
        """Test types imported from both modules are the same objects."""
        from backend.execution.broker_base import BrokerInterface as BaseBrokerInterface
        from backend.execution.broker_base import Order as BaseOrder
        from backend.execution.broker_interface import (
            BrokerInterface as InterfaceBrokerInterface,
        )
        from backend.execution.broker_interface import Order as InterfaceOrder

        assert BaseBrokerInterface is InterfaceBrokerInterface
        assert BaseOrder is InterfaceOrder


# Test dataclass defaults and types
class TestDataclassTypes:
    """Tests for dataclass type correctness."""

    def test_order_enum_types(self):
        """Test Order uses correct enum types."""
        order = Order(
            order_id="TEST",
            symbol="SPY",
            qty=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        assert isinstance(order.side, OrderSide)
        assert isinstance(order.order_type, OrderType)
        assert isinstance(order.status, OrderStatus)

    def test_position_calculated_fields(self):
        """Test Position calculates derived fields in __post_init__."""
        position = Position(
            symbol="SPY",
            qty=100,
            entry_price=450.0,
            current_price=460.0,
        )

        assert position.market_value == 46000.0
        assert position.unrealized_pnl == 1000.0

    def test_account_default_positions_value(self):
        """Test Account has default positions_value of 0."""
        account = Account(
            account_id="TEST",
            balance=100000.0,
            buying_power=100000.0,
            equity=100000.0,
            cash=100000.0,
        )

        assert account.positions_value == 0.0


# Test new interface requirements satisfaction
class TestInterfaceRequirements:
    """Tests verifying interface meets Phase A requirements."""

    def test_interface_has_connect_method(self):
        """Test BrokerInterface defines connect() method."""
        assert hasattr(BrokerInterface, "connect")
        assert callable(getattr(BrokerInterface, "connect"))

    def test_interface_has_disconnect_method(self):
        """Test BrokerInterface defines disconnect() method."""
        assert hasattr(BrokerInterface, "disconnect")
        assert callable(getattr(BrokerInterface, "disconnect"))

    def test_interface_has_health_check_method(self):
        """Test BrokerInterface defines health_check() method."""
        assert hasattr(BrokerInterface, "health_check")
        assert callable(getattr(BrokerInterface, "health_check"))

    def test_interface_has_get_account_method(self):
        """Test BrokerInterface defines get_account() method."""
        assert hasattr(BrokerInterface, "get_account")
        assert callable(getattr(BrokerInterface, "get_account"))

    def test_interface_has_get_positions_method(self):
        """Test BrokerInterface defines get_positions() method."""
        assert hasattr(BrokerInterface, "get_positions")
        assert callable(getattr(BrokerInterface, "get_positions"))

    def test_interface_has_place_order_method(self):
        """Test BrokerInterface defines place_order() method."""
        assert hasattr(BrokerInterface, "place_order")
        assert callable(getattr(BrokerInterface, "place_order"))

    def test_interface_has_cancel_order_method(self):
        """Test BrokerInterface defines cancel_order() method."""
        assert hasattr(BrokerInterface, "cancel_order")
        assert callable(getattr(BrokerInterface, "cancel_order"))

    def test_interface_has_get_order_status_method(self):
        """Test BrokerInterface defines get_order_status() method."""
        assert hasattr(BrokerInterface, "get_order_status")
        assert callable(getattr(BrokerInterface, "get_order_status"))

    @pytest.mark.asyncio
    async def test_paper_broker_implements_all_methods(self):
        """Test PaperBroker implements all BrokerInterface methods."""
        broker = PaperBroker()
        broker.set_market_price("SPY", 450.0)

        # Connection lifecycle
        connect_result = await broker.connect()
        assert isinstance(connect_result, bool)

        health = await broker.health_check()
        assert isinstance(health, BrokerHealth)

        # Account and positions
        account = await broker.get_account()
        assert isinstance(account, Account)

        positions = await broker.get_positions()
        assert isinstance(positions, list)

        # Order operations
        order = await broker.place_order(
            symbol="SPY",
            qty=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        assert isinstance(order, Order)

        status = await broker.get_order_status(order.order_id)
        assert isinstance(status, Order)

        limit_order = await broker.place_order(
            symbol="SPY",
            qty=50,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=460.0,
        )
        cancel_result = await broker.cancel_order(limit_order.order_id)
        assert isinstance(cancel_result, bool)

        # Disconnect
        await broker.disconnect()
        assert broker._connected is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
