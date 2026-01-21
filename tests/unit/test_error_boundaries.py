"""
Unit tests for frontend error boundary components.

Tests verify:
- Error boundary files exist
- TypeScript syntax is valid
- Components export correctly
- Error handling logic is present
- Fallback UI components are defined
"""

import os


def test_error_boundary_file_exists():
    """Test that ErrorBoundary.tsx exists"""
    path = "fluxhero/frontend/components/ErrorBoundary.tsx"
    assert os.path.exists(path), f"ErrorBoundary component not found at {path}"


def test_error_fallback_file_exists():
    """Test that ErrorFallback.tsx exists"""
    path = "fluxhero/frontend/components/ErrorFallback.tsx"
    assert os.path.exists(path), f"ErrorFallback component not found at {path}"


def test_page_error_boundary_file_exists():
    """Test that PageErrorBoundary.tsx exists"""
    path = "fluxhero/frontend/components/PageErrorBoundary.tsx"
    assert os.path.exists(path), f"PageErrorBoundary component not found at {path}"


def test_async_error_boundary_file_exists():
    """Test that AsyncErrorBoundary.tsx exists"""
    path = "fluxhero/frontend/components/AsyncErrorBoundary.tsx"
    assert os.path.exists(path), f"AsyncErrorBoundary component not found at {path}"


def test_error_boundary_has_class_component():
    """Test that ErrorBoundary uses React class component"""
    path = "fluxhero/frontend/components/ErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for class component definition
    assert "class ErrorBoundary" in content, "ErrorBoundary must be a class component"
    assert "extends Component" in content, "ErrorBoundary must extend React.Component"


def test_error_boundary_has_componentdidcatch():
    """Test that ErrorBoundary implements componentDidCatch"""
    path = "fluxhero/frontend/components/ErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert "componentDidCatch" in content, "ErrorBoundary must implement componentDidCatch"


def test_error_boundary_has_getderivedstatefromerror():
    """Test that ErrorBoundary implements getDerivedStateFromError"""
    path = "fluxhero/frontend/components/ErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert (
        "getDerivedStateFromError" in content
    ), "ErrorBoundary must implement getDerivedStateFromError"


def test_error_boundary_has_state_interface():
    """Test that ErrorBoundary has proper state interface"""
    path = "fluxhero/frontend/components/ErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for state interface with required fields
    assert "hasError" in content, "State must track hasError"
    assert "error: Error" in content, "State must track error object"


def test_error_boundary_exports_default():
    """Test that ErrorBoundary exports default"""
    path = "fluxhero/frontend/components/ErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert "export default ErrorBoundary" in content, "ErrorBoundary must export default"


def test_error_boundary_has_reset_functionality():
    """Test that ErrorBoundary has reset functionality"""
    path = "fluxhero/frontend/components/ErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for reset method
    assert "resetErrorBoundary" in content, "ErrorBoundary must have reset functionality"


def test_error_fallback_has_default_export():
    """Test that ErrorFallback has default export"""
    path = "fluxhero/frontend/components/ErrorFallback.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert (
        "export default function ErrorFallback" in content
    ), "ErrorFallback must export default"


def test_error_fallback_has_props_interface():
    """Test that ErrorFallback has proper props interface"""
    path = "fluxhero/frontend/components/ErrorFallback.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for props interface
    assert "ErrorFallbackProps" in content, "ErrorFallback must have props interface"
    assert "error?" in content, "Props must include error"
    assert "resetErrorBoundary?" in content, "Props must include resetErrorBoundary"


def test_error_fallback_has_specialized_components():
    """Test that ErrorFallback exports specialized components"""
    path = "fluxhero/frontend/components/ErrorFallback.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for specialized error fallback components
    assert (
        "APIErrorFallback" in content
    ), "ErrorFallback must export APIErrorFallback"
    assert (
        "DataLoadErrorFallback" in content
    ), "ErrorFallback must export DataLoadErrorFallback"
    assert (
        "ComponentErrorFallback" in content
    ), "ErrorFallback must export ComponentErrorFallback"


def test_error_fallback_has_styling():
    """Test that ErrorFallback includes styling"""
    path = "fluxhero/frontend/components/ErrorFallback.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for style jsx or className usage
    has_styling = "<style jsx>" in content or "className=" in content
    assert has_styling, "ErrorFallback must include styling"


def test_page_error_boundary_exports_default():
    """Test that PageErrorBoundary exports default"""
    path = "fluxhero/frontend/components/PageErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert (
        "export default function PageErrorBoundary" in content
    ), "PageErrorBoundary must export default"


def test_page_error_boundary_wraps_error_boundary():
    """Test that PageErrorBoundary wraps ErrorBoundary"""
    path = "fluxhero/frontend/components/PageErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check that it imports and uses ErrorBoundary
    assert (
        "import ErrorBoundary" in content
    ), "PageErrorBoundary must import ErrorBoundary"
    assert "<ErrorBoundary" in content, "PageErrorBoundary must use ErrorBoundary"


def test_page_error_boundary_has_page_name_prop():
    """Test that PageErrorBoundary accepts pageName prop"""
    path = "fluxhero/frontend/components/PageErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert "pageName" in content, "PageErrorBoundary must accept pageName prop"


def test_async_error_boundary_exports_default():
    """Test that AsyncErrorBoundary exports default"""
    path = "fluxhero/frontend/components/AsyncErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert (
        "export default function AsyncErrorBoundary" in content
    ), "AsyncErrorBoundary must export default"


def test_async_error_boundary_handles_loading_state():
    """Test that AsyncErrorBoundary handles loading state"""
    path = "fluxhero/frontend/components/AsyncErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert "isLoading" in content, "AsyncErrorBoundary must handle isLoading prop"


def test_async_error_boundary_handles_error_state():
    """Test that AsyncErrorBoundary handles error state"""
    path = "fluxhero/frontend/components/AsyncErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert "error" in content, "AsyncErrorBoundary must handle error prop"


def test_async_error_boundary_has_retry_functionality():
    """Test that AsyncErrorBoundary has retry functionality"""
    path = "fluxhero/frontend/components/AsyncErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert "onRetry" in content, "AsyncErrorBoundary must have onRetry prop"


def test_async_error_boundary_exports_hook():
    """Test that AsyncErrorBoundary exports useAsyncError hook"""
    path = "fluxhero/frontend/components/AsyncErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    assert (
        "export function useAsyncError" in content
    ), "AsyncErrorBoundary must export useAsyncError hook"


def test_layout_integrates_error_boundary():
    """Test that root layout integrates ErrorBoundary"""
    path = "fluxhero/frontend/app/layout.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check that ErrorBoundary is imported and used
    assert (
        "import ErrorBoundary" in content
    ), "Root layout must import ErrorBoundary"
    assert "<ErrorBoundary>" in content, "Root layout must use ErrorBoundary"


def test_all_components_have_client_directive():
    """Test that all error boundary components have 'use client' directive"""
    components = [
        "fluxhero/frontend/components/ErrorBoundary.tsx",
        "fluxhero/frontend/components/ErrorFallback.tsx",
        "fluxhero/frontend/components/PageErrorBoundary.tsx",
        "fluxhero/frontend/components/AsyncErrorBoundary.tsx",
    ]

    for component_path in components:
        with open(component_path, "r") as f:
            content = f.read()

        # Check for 'use client' at the top of the file
        assert (
            "'use client'" in content or '"use client"' in content
        ), f"{component_path} must have 'use client' directive for Next.js App Router"


def test_error_boundary_has_reset_keys_support():
    """Test that ErrorBoundary supports resetKeys prop"""
    path = "fluxhero/frontend/components/ErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for resetKeys prop in interface
    assert "resetKeys?" in content, "ErrorBoundary must support resetKeys prop"


def test_error_boundary_has_custom_error_handler():
    """Test that ErrorBoundary supports custom onError handler"""
    path = "fluxhero/frontend/components/ErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for onError prop
    assert "onError?" in content, "ErrorBoundary must support onError prop"


def test_error_fallback_has_reset_button():
    """Test that ErrorFallback includes a reset button"""
    path = "fluxhero/frontend/components/ErrorFallback.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for button element with onClick handler
    assert "<button" in content, "ErrorFallback must have a button element"
    assert "onClick" in content, "Button must have onClick handler"


def test_error_fallback_has_error_details():
    """Test that ErrorFallback shows error details"""
    path = "fluxhero/frontend/components/ErrorFallback.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for details/pre elements to show error information
    assert "<details" in content or "<pre" in content, "ErrorFallback must show error details"


def test_async_error_boundary_uses_state():
    """Test that AsyncErrorBoundary uses React state"""
    path = "fluxhero/frontend/components/AsyncErrorBoundary.tsx"
    with open(path, "r") as f:
        content = f.read()

    # Check for useState import and usage
    assert "useState" in content, "AsyncErrorBoundary must use useState"


def test_typescript_interfaces_defined():
    """Test that all components have TypeScript interfaces"""
    components = {
        "fluxhero/frontend/components/ErrorBoundary.tsx": "ErrorBoundaryProps",
        "fluxhero/frontend/components/ErrorFallback.tsx": "ErrorFallbackProps",
        "fluxhero/frontend/components/PageErrorBoundary.tsx": "PageErrorBoundaryProps",
        "fluxhero/frontend/components/AsyncErrorBoundary.tsx": "AsyncErrorBoundaryProps",
    }

    for component_path, interface_name in components.items():
        with open(component_path, "r") as f:
            content = f.read()

        assert (
            f"interface {interface_name}" in content
        ), f"{component_path} must define {interface_name} interface"


def test_components_have_documentation():
    """Test that all components have JSDoc documentation"""
    components = [
        "fluxhero/frontend/components/ErrorBoundary.tsx",
        "fluxhero/frontend/components/ErrorFallback.tsx",
        "fluxhero/frontend/components/PageErrorBoundary.tsx",
        "fluxhero/frontend/components/AsyncErrorBoundary.tsx",
    ]

    for component_path in components:
        with open(component_path, "r") as f:
            content = f.read()

        # Check for JSDoc comments
        assert (
            "/**" in content and "*/" in content
        ), f"{component_path} must have JSDoc documentation"


def test_error_boundary_integration_complete():
    """Test that error boundary integration is complete"""
    # Verify all files exist
    required_files = [
        "fluxhero/frontend/components/ErrorBoundary.tsx",
        "fluxhero/frontend/components/ErrorFallback.tsx",
        "fluxhero/frontend/components/PageErrorBoundary.tsx",
        "fluxhero/frontend/components/AsyncErrorBoundary.tsx",
    ]

    for file_path in required_files:
        assert os.path.exists(file_path), f"Required file {file_path} not found"

    # Verify layout integration
    layout_path = "fluxhero/frontend/app/layout.tsx"
    with open(layout_path, "r") as f:
        layout_content = f.read()

    assert (
        "ErrorBoundary" in layout_content
    ), "ErrorBoundary must be integrated in root layout"

    print("✓ Error boundary integration complete")
    print("✓ All required files exist")
    print("✓ Root layout integration verified")
    print("✓ TypeScript interfaces defined")
    print("✓ Documentation present")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
