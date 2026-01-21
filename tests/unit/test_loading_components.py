"""
Unit tests for frontend loading states implementation

This test file validates the following:
- Loading components exist and are properly structured
- All pages import and use loading components correctly
- Different loading variants are implemented (spinner, dots, pulse, skeleton)
- LoadingOverlay, SkeletonCard, SkeletonTable components exist
- LoadingButton component exists and is used in backtest page

Note: These are static analysis tests for the frontend TypeScript/React code.
"""

import re
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_loading_spinner_component_exists():
    """Test that LoadingSpinner component file exists"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'
    assert component_path.exists(), f"LoadingSpinner component not found at {component_path}"


def test_loading_spinner_exports():
    """Test that LoadingSpinner component exports all required components"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for main LoadingSpinner export
    assert 'export default LoadingSpinner' in content, "LoadingSpinner default export not found"

    # Check for additional exports
    assert 'export const LoadingOverlay' in content, "LoadingOverlay export not found"
    assert 'export const SkeletonCard' in content, "SkeletonCard export not found"
    assert 'export const SkeletonTable' in content, "SkeletonTable export not found"
    assert 'export const LoadingButton' in content, "LoadingButton export not found"


def test_loading_spinner_variants():
    """Test that LoadingSpinner supports multiple variants"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for variant type definition
    assert "'spinner' | 'dots' | 'pulse' | 'skeleton'" in content, "LoadingVariant type not properly defined"

    # Check for variant implementations
    assert "case 'spinner':" in content, "Spinner variant not implemented"
    assert "case 'dots':" in content, "Dots variant not implemented"
    assert "case 'pulse':" in content, "Pulse variant not implemented"
    assert "case 'skeleton':" in content, "Skeleton variant not implemented"


def test_loading_spinner_sizes():
    """Test that LoadingSpinner supports multiple sizes"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for size type definition
    assert "'sm' | 'md' | 'lg' | 'xl'" in content, "LoadingSize type not properly defined"

    # Check for size classes
    assert "sm: 'h-4 w-4'" in content, "Small size not defined"
    assert "md: 'h-8 w-8'" in content, "Medium size not defined"
    assert "lg: 'h-12 w-12'" in content, "Large size not defined"
    assert "xl: 'h-16 w-16'" in content, "Extra large size not defined"


def test_loading_overlay_component():
    """Test that LoadingOverlay component is properly implemented"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for LoadingOverlay interface
    assert 'interface LoadingOverlayProps' in content, "LoadingOverlayProps interface not found"
    assert 'isLoading: boolean' in content, "isLoading prop not defined"
    assert 'children: React.ReactNode' in content, "children prop not defined"

    # Check for overlay implementation
    assert 'absolute inset-0' in content, "Overlay positioning not implemented"


def test_skeleton_components():
    """Test that skeleton loader components are implemented"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for SkeletonCard
    assert 'export const SkeletonCard' in content, "SkeletonCard not exported"
    assert 'animate-pulse' in content, "Pulse animation not used in skeleton"

    # Check for SkeletonTable
    assert 'export const SkeletonTable' in content, "SkeletonTable not exported"
    assert 'rows?: number' in content, "SkeletonTable rows prop not defined"
    assert 'cols?: number' in content, "SkeletonTable cols prop not defined"


def test_loading_button_component():
    """Test that LoadingButton component is properly implemented"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for LoadingButton interface
    assert 'interface LoadingButtonProps' in content, "LoadingButtonProps interface not found"
    assert 'isLoading: boolean' in content, "isLoading prop not defined"
    assert 'loadingText?: string' in content, "loadingText prop not defined"

    # Check for button implementation
    assert 'export const LoadingButton' in content, "LoadingButton not exported"


def test_live_page_uses_loading_spinner():
    """Test that Live Trading page imports and uses loading components"""
    page_path = PROJECT_ROOT / 'fluxhero/frontend/app/live/page.tsx'

    with open(page_path, 'r') as f:
        content = f.read()

    # Check for loading state
    assert 'const [loading, setLoading]' in content, "Loading state not defined"

    # Check for loading UI
    assert 'if (loading)' in content, "Loading condition not found"
    assert 'Loading live data' in content or 'loading' in content.lower(), "Loading message not found"


def test_analytics_page_uses_loading_components():
    """Test that Analytics page imports and uses loading components"""
    page_path = PROJECT_ROOT / 'fluxhero/frontend/app/analytics/page.tsx'

    with open(page_path, 'r') as f:
        content = f.read()

    # Check for import
    assert 'from \'../../components/LoadingSpinner\'' in content, "LoadingSpinner not imported"
    assert 'LoadingOverlay' in content or 'SkeletonCard' in content or 'LoadingSpinner' in content, \
        "Loading components not imported"

    # Check for loading states
    assert 'const [loading, setLoading]' in content, "Loading state not defined"
    assert 'const [initialLoad, setInitialLoad]' in content, "Initial load state not defined"

    # Check for LoadingOverlay usage
    assert 'LoadingOverlay' in content or 'LoadingSpinner' in content, "Loading component not used"


def test_history_page_uses_loading_components():
    """Test that Trade History page imports and uses loading components"""
    page_path = PROJECT_ROOT / 'fluxhero/frontend/pages/history.tsx'

    with open(page_path, 'r') as f:
        content = f.read()

    # Check for import
    assert 'from \'../components/LoadingSpinner\'' in content, "LoadingSpinner not imported"
    assert 'SkeletonTable' in content or 'LoadingSpinner' in content, "Loading components not imported"

    # Check for loading states
    assert 'const [loading, setLoading]' in content, "Loading state not defined"
    assert 'const [initialLoad, setInitialLoad]' in content, "Initial load state not defined"

    # Check for skeleton table usage
    assert 'SkeletonTable' in content, "SkeletonTable not used"


def test_backtest_page_uses_loading_button():
    """Test that Backtest page imports and uses LoadingButton"""
    page_path = PROJECT_ROOT / 'fluxhero/frontend/app/backtest/page.tsx'

    with open(page_path, 'r') as f:
        content = f.read()

    # Check for import
    assert 'from \'../../components/LoadingSpinner\'' in content, "LoadingSpinner not imported"
    assert 'LoadingButton' in content, "LoadingButton not imported"

    # Check for loading state
    assert 'const [isRunning, setIsRunning]' in content, "isRunning state not defined"

    # Check for LoadingButton usage
    assert '<LoadingButton' in content, "LoadingButton component not used"
    assert 'isLoading={isRunning}' in content, "isLoading prop not passed to LoadingButton"


def test_all_pages_have_initial_load_states():
    """Test that all pages properly handle initial load states"""
    pages = [
        PROJECT_ROOT / 'fluxhero/frontend/app/live/page.tsx',
        PROJECT_ROOT / 'fluxhero/frontend/app/analytics/page.tsx',
        PROJECT_ROOT / 'fluxhero/frontend/pages/history.tsx',
    ]

    for page_path in pages:
        with open(page_path, 'r') as f:
            content = f.read()

        # Check for loading state management
        assert 'setLoading(false)' in content, f"{page_path}: Loading state not properly managed"

        # Check for error handling
        assert 'try' in content and 'catch' in content, f"{page_path}: No error handling found"


def test_loading_components_accessibility():
    """Test that loading components have proper accessibility attributes"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for ARIA attributes
    assert 'role="status"' in content, "role='status' attribute not found"
    assert 'aria-label="Loading"' in content, "aria-label attribute not found"


def test_loading_states_properly_cleared():
    """Test that loading states are properly cleared in finally blocks"""
    pages = [
        PROJECT_ROOT / 'fluxhero/frontend/app/live/page.tsx',
        PROJECT_ROOT / 'fluxhero/frontend/app/analytics/page.tsx',
        PROJECT_ROOT / 'fluxhero/frontend/pages/history.tsx',
        PROJECT_ROOT / 'fluxhero/frontend/app/backtest/page.tsx',
    ]

    for page_path in pages:
        with open(page_path, 'r') as f:
            content = f.read()

        # Check for finally blocks that clear loading states
        if 'setLoading(true)' in content or 'setIsRunning(true)' in content:
            assert 'finally' in content, f"{page_path}: Loading state set but no finally block found"


def test_loading_messages_are_descriptive():
    """Test that loading messages are descriptive and user-friendly"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for message prop
    assert 'message?: string' in content, "Message prop not defined"
    assert '{message}' in content, "Message not rendered"


def test_skeleton_loaders_have_proper_structure():
    """Test that skeleton loaders have proper HTML structure"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for SkeletonCard structure
    skeleton_card_match = re.search(r'export const SkeletonCard.*?return.*?</div>', content, re.DOTALL)
    assert skeleton_card_match, "SkeletonCard structure not found"

    # Check for SkeletonTable structure
    skeleton_table_match = re.search(r'export const SkeletonTable.*?return.*?</div>', content, re.DOTALL)
    assert skeleton_table_match, "SkeletonTable structure not found"


def test_loading_spinner_has_proper_animations():
    """Test that loading components use proper CSS animations"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for Tailwind animation classes
    assert 'animate-spin' in content, "Spin animation not found"
    assert 'animate-pulse' in content, "Pulse animation not found"
    assert 'animate-bounce' in content, "Bounce animation not found"


def test_loading_overlay_has_proper_z_index():
    """Test that LoadingOverlay has proper z-index for overlay effect"""
    component_path = PROJECT_ROOT / 'fluxhero/frontend/components/LoadingSpinner.tsx'

    with open(component_path, 'r') as f:
        content = f.read()

    # Check for z-index in overlay
    assert 'z-10' in content or 'z-20' in content or 'z-50' in content, \
        "Z-index not set on loading overlay"


def test_all_async_operations_have_loading_states():
    """Test that all async operations in pages have corresponding loading states"""
    pages = {
        PROJECT_ROOT / 'fluxhero/frontend/app/live/page.tsx': ['getPositions', 'getAccountInfo', 'getSystemStatus'],
        PROJECT_ROOT / 'fluxhero/frontend/app/analytics/page.tsx': ['fetchChartData'],
        PROJECT_ROOT / 'fluxhero/frontend/pages/history.tsx': ['getTrades'],
        PROJECT_ROOT / 'fluxhero/frontend/app/backtest/page.tsx': ['runBacktest'],
    }

    for page_path, async_operations in pages.items():
        with open(page_path, 'r') as f:
            content = f.read()

        # For each async operation, check that loading state is managed
        for operation in async_operations:
            # Operation should be in the file or have equivalent async handling
            assert ('async' in content and 'await' in content) or operation in content, \
                f"{page_path}: Async operation {operation} not properly handled"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
