"""
Unit tests for frontend WebSocket implementation
Tests the React hook, context provider, and status components
"""

import os

import pytest


class TestWebSocketHook:
    """Tests for useWebSocket React hook"""

    def test_hook_file_exists(self):
        """Test that useWebSocket hook file exists"""
        hook_path = "frontend/hooks/useWebSocket.ts"
        assert os.path.exists(hook_path), f"Hook file not found: {hook_path}"

    def test_hook_exports_websocket_state_enum(self):
        """Test that hook exports WebSocketState enum"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "export enum WebSocketState" in content
            assert "CONNECTING = 'CONNECTING'" in content
            assert "CONNECTED = 'CONNECTED'" in content
            assert "DISCONNECTED = 'DISCONNECTED'" in content
            assert "RECONNECTING = 'RECONNECTING'" in content
            assert "FAILED = 'FAILED'" in content

    def test_hook_exports_price_update_interface(self):
        """Test that hook exports PriceUpdate interface"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "export interface PriceUpdate" in content
            assert "symbol: string" in content
            assert "price: number" in content
            assert "timestamp: string" in content

    def test_hook_exports_websocket_options(self):
        """Test that hook exports WebSocketOptions interface"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "export interface WebSocketOptions" in content
            assert "autoReconnect?: boolean" in content
            assert "maxReconnectAttempts?: number" in content
            assert "reconnectDelay?: number" in content

    def test_hook_exports_use_websocket_return_interface(self):
        """Test that hook exports return type interface"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "export interface UseWebSocketReturn" in content
            assert "state: WebSocketState" in content
            assert "data: PriceUpdate | null" in content
            assert "error: string | null" in content
            assert "reconnect: () => void" in content
            assert "disconnect: () => void" in content
            assert "send: (message: string | object) => void" in content

    def test_hook_exports_use_websocket_function(self):
        """Test that hook exports useWebSocket function"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "export function useWebSocket" in content
            assert "url: string" in content
            assert "options: WebSocketOptions = {}" in content
            assert "UseWebSocketReturn" in content

    def test_hook_has_reconnection_logic(self):
        """Test that hook implements reconnection logic"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "getReconnectDelay" in content
            assert "Math.pow(2, attempt)" in content  # Exponential backoff
            assert "reconnectTimeoutRef" in content
            assert "shouldReconnectRef" in content

    def test_hook_has_connection_management(self):
        """Test that hook has connect/disconnect methods"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "const connect = useCallback" in content
            assert "const disconnect = useCallback" in content
            assert "ws.onopen" in content
            assert "ws.onmessage" in content
            assert "ws.onerror" in content
            assert "ws.onclose" in content

    def test_hook_has_send_method(self):
        """Test that hook has send method for messages"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "const send = useCallback" in content
            assert "wsRef.current.send" in content
            assert "JSON.stringify(message)" in content

    def test_hook_has_lifecycle_management(self):
        """Test that hook manages lifecycle with useEffect"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "useEffect(() => {" in content
            assert "connect();" in content
            assert "return () => {" in content  # Cleanup function
            assert "wsRef.current.close()" in content


class TestWebSocketContext:
    """Tests for WebSocket context provider"""

    def test_context_file_exists(self):
        """Test that context file exists"""
        context_path = "frontend/contexts/WebSocketContext.tsx"
        assert os.path.exists(context_path), f"Context file not found: {context_path}"

    def test_context_exports_provider(self):
        """Test that context exports WebSocketProvider"""
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            content = f.read()
            assert "export function WebSocketProvider" in content
            assert "children: React.ReactNode" in content

    def test_context_exports_hook(self):
        """Test that context exports useWebSocketContext hook"""
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            content = f.read()
            assert "export function useWebSocketContext" in content
            assert "useContext(WebSocketContext)" in content

    def test_context_value_interface(self):
        """Test that context defines proper value interface"""
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            content = f.read()
            assert "interface WebSocketContextValue" in content
            assert "connectionState: WebSocketState" in content
            assert "prices: PriceMap" in content
            assert "getPrice: (symbol: string)" in content
            assert "subscribe: (symbols: string[])" in content
            assert "unsubscribe: (symbols: string[])" in content
            assert "reconnect: () => void" in content

    def test_context_has_price_map(self):
        """Test that context manages price map"""
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            content = f.read()
            assert "interface PriceMap" in content
            assert "[symbol: string]: PriceUpdate" in content
            assert "const [prices, setPrices] = useState<PriceMap>" in content

    def test_context_has_subscribe_unsubscribe(self):
        """Test that context has subscribe/unsubscribe methods"""
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            content = f.read()
            assert "const subscribe = useCallback" in content
            assert "const unsubscribe = useCallback" in content
            assert "action: 'subscribe'" in content
            assert "action: 'unsubscribe'" in content

    def test_context_handles_reconnection(self):
        """Test that context handles reconnection with re-subscription"""
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            content = f.read()
            assert "onOpen: () => {" in content
            assert "Re-subscribe to symbols on reconnect" in content
            assert "subscribedSymbols" in content

    def test_context_has_client_directive(self):
        """Test that context has 'use client' directive for Next.js"""
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            content = f.read()
            assert "'use client'" in content or '"use client"' in content


class TestWebSocketStatusComponent:
    """Tests for WebSocket status indicator components"""

    def test_status_component_file_exists(self):
        """Test that status component file exists"""
        component_path = "frontend/components/WebSocketStatus.tsx"
        assert os.path.exists(component_path), f"Component file not found: {component_path}"

    def test_status_component_exports_main_component(self):
        """Test that component exports WebSocketStatus"""
        with open("frontend/components/WebSocketStatus.tsx") as f:
            content = f.read()
            assert "export function WebSocketStatus" in content
            assert "useWebSocketContext" in content

    def test_status_component_exports_badge(self):
        """Test that component exports compact badge version"""
        with open("frontend/components/WebSocketStatus.tsx") as f:
            content = f.read()
            assert "export function WebSocketStatusBadge" in content

    def test_status_component_has_props_interface(self):
        """Test that component defines props interface"""
        with open("frontend/components/WebSocketStatus.tsx") as f:
            content = f.read()
            assert "interface WebSocketStatusProps" in content
            assert "showText?: boolean" in content
            assert "className?: string" in content

    def test_status_component_uses_websocket_context(self):
        """Test that component uses WebSocket context"""
        with open("frontend/components/WebSocketStatus.tsx") as f:
            content = f.read()
            assert "const { connectionState, error, reconnectAttempts, reconnect }" in content

    def test_status_component_has_status_mapping(self):
        """Test that component maps connection states to visual indicators"""
        with open("frontend/components/WebSocketStatus.tsx") as f:
            content = f.read()
            assert "getStatusConfig" in content
            assert "WebSocketState.CONNECTED" in content
            assert "WebSocketState.CONNECTING" in content
            assert "WebSocketState.RECONNECTING" in content
            assert "WebSocketState.DISCONNECTED" in content
            assert "WebSocketState.FAILED" in content

    def test_status_component_has_retry_button(self):
        """Test that component shows retry button on failure"""
        with open("frontend/components/WebSocketStatus.tsx") as f:
            content = f.read()
            assert "showRetry" in content
            assert "onClick={reconnect}" in content
            assert "Retry" in content

    def test_status_component_has_color_coding(self):
        """Test that component uses color-coded indicators"""
        with open("frontend/components/WebSocketStatus.tsx") as f:
            content = f.read()
            assert "bg-green-500" in content  # Connected
            assert "bg-yellow-500" in content  # Connecting
            assert "bg-red-500" in content  # Failed
            assert "bg-orange-500" in content  # Reconnecting
            assert "bg-gray-500" in content  # Disconnected

    def test_status_badge_has_pulse_animation(self):
        """Test that badge uses pulse animation for connecting states"""
        with open("frontend/components/WebSocketStatus.tsx") as f:
            content = f.read()
            assert "animate-pulse" in content
            assert "pulse: true" in content or "pulse: false" in content

    def test_status_component_has_client_directive(self):
        """Test that component has 'use client' directive for Next.js"""
        with open("frontend/components/WebSocketStatus.tsx") as f:
            content = f.read()
            assert "'use client'" in content or '"use client"' in content


class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality"""

    def test_hooks_directory_exists(self):
        """Test that hooks directory exists"""
        assert os.path.exists("frontend/hooks")

    def test_contexts_directory_exists(self):
        """Test that contexts directory exists"""
        assert os.path.exists("frontend/contexts")

    def test_components_directory_exists(self):
        """Test that components directory exists"""
        assert os.path.exists("frontend/components")

    def test_all_websocket_files_exist(self):
        """Test that all WebSocket-related files exist"""
        files = [
            "frontend/hooks/useWebSocket.ts",
            "frontend/contexts/WebSocketContext.tsx",
            "frontend/components/WebSocketStatus.tsx",
        ]
        for file_path in files:
            assert os.path.exists(file_path), f"File not found: {file_path}"

    def test_imports_are_consistent(self):
        """Test that imports between files are consistent"""
        # Check that context imports from hook
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            context_content = f.read()
            assert "from '../hooks/useWebSocket'" in context_content
            assert "useWebSocket" in context_content
            assert "WebSocketState" in context_content

        # Check that component imports from context
        with open("frontend/components/WebSocketStatus.tsx") as f:
            component_content = f.read()
            assert "from '../contexts/WebSocketContext'" in component_content
            assert "useWebSocketContext" in component_content

    def test_websocket_url_configuration(self):
        """Test that WebSocket URL is configurable"""
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            content = f.read()
            assert "url = '/ws/prices'" in content or 'url = "/ws/prices"' in content

    def test_reconnection_parameters_configurable(self):
        """Test that reconnection parameters are configurable"""
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            content = f.read()
            assert "autoReconnect: true" in content
            assert "maxReconnectAttempts: 5" in content
            assert "reconnectDelay: 1000" in content

    def test_exponential_backoff_implemented(self):
        """Test that exponential backoff is implemented"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "Math.pow(2, attempt)" in content
            assert "maxReconnectDelay" in content

    def test_message_parsing_implemented(self):
        """Test that message parsing is implemented"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "JSON.parse(event.data)" in content
            assert "catch" in content  # Error handling for parse failures


class TestWebSocketDocumentation:
    """Tests for WebSocket documentation and examples"""

    def test_hook_has_jsdoc_comments(self):
        """Test that hook has comprehensive JSDoc comments"""
        with open("frontend/hooks/useWebSocket.ts") as f:
            content = f.read()
            assert "/**" in content
            assert "@param" in content
            assert "@returns" in content
            assert "@example" in content

    def test_context_has_usage_examples(self):
        """Test that context has usage examples"""
        with open("frontend/contexts/WebSocketContext.tsx") as f:
            content = f.read()
            assert "@example" in content
            assert "useWebSocketContext" in content

    def test_component_has_usage_examples(self):
        """Test that component has usage examples"""
        with open("frontend/components/WebSocketStatus.tsx") as f:
            content = f.read()
            assert "@example" in content or "Example" in content

    def test_all_files_have_file_headers(self):
        """Test that all files have descriptive headers"""
        files = [
            "frontend/hooks/useWebSocket.ts",
            "frontend/contexts/WebSocketContext.tsx",
            "frontend/components/WebSocketStatus.tsx",
        ]
        for file_path in files:
            with open(file_path) as f:
                first_lines = "".join(f.readline() for _ in range(5))
                assert "/**" in first_lines or "//" in first_lines, (
                    f"File {file_path} missing header comment"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
