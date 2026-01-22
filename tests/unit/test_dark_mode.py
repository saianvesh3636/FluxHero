"""
Unit tests for dark mode functionality.

Tests the theme context, localStorage persistence, and theme toggle component.
"""

import os
import pytest

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_project_path(relative_path):
    """Get absolute path from project root."""
    return os.path.join(PROJECT_ROOT, relative_path)


class TestDarkModeImplementation:
    """Test suite for dark mode implementation."""

    def test_theme_context_file_exists(self):
        """Test that theme context file exists."""
        path = get_project_path("frontend/utils/theme-context.tsx")
        assert os.path.exists(path), f"Theme context file not found: {path}"

    def test_theme_toggle_component_exists(self):
        """Test that theme toggle component exists."""
        path = get_project_path("frontend/components/ThemeToggle.tsx")
        assert os.path.exists(path), f"Theme toggle component not found: {path}"

    def test_theme_context_has_provider(self):
        """Test that theme context exports ThemeProvider."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "ThemeProvider" in content, "ThemeProvider not found in theme context"
            assert "export function ThemeProvider" in content, "ThemeProvider not exported"

    def test_theme_context_has_hook(self):
        """Test that theme context exports useTheme hook."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "useTheme" in content, "useTheme hook not found"
            assert "export function useTheme" in content, "useTheme hook not exported"

    def test_theme_context_has_toggle_function(self):
        """Test that theme context has toggleTheme function."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "toggleTheme" in content, "toggleTheme function not found"

    def test_theme_context_uses_localstorage(self):
        """Test that theme context uses localStorage for persistence."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "localStorage" in content, "localStorage not used for persistence"
            assert "getItem" in content, "localStorage.getItem not used"
            assert "setItem" in content, "localStorage.setItem not used"

    def test_theme_context_has_storage_key(self):
        """Test that theme context defines a storage key."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "STORAGE_KEY" in content or "fluxhero-theme" in content, "Storage key not defined"

    def test_theme_toggle_imports_hook(self):
        """Test that theme toggle component imports useTheme hook."""
        with open(get_project_path("frontend/components/ThemeToggle.tsx"), "r") as f:
            content = f.read()
            assert "useTheme" in content, "useTheme hook not imported in ThemeToggle"
            assert "from '../utils/theme-context'" in content, "Theme context not imported"

    def test_theme_toggle_has_button(self):
        """Test that theme toggle component has a button element."""
        with open(get_project_path("frontend/components/ThemeToggle.tsx"), "r") as f:
            content = f.read()
            assert "<button" in content, "Button element not found in ThemeToggle"
            assert "onClick" in content, "onClick handler not found"

    def test_theme_toggle_has_icons(self):
        """Test that theme toggle component has light and dark mode icons."""
        with open(get_project_path("frontend/components/ThemeToggle.tsx"), "r") as f:
            content = f.read()
            assert "<svg" in content, "SVG icons not found"
            assert content.count("<svg") >= 2, "Both light and dark mode icons should be present"

    def test_layout_includes_theme_provider(self):
        """Test that root layout includes ThemeProvider."""
        with open(get_project_path("frontend/app/layout.tsx"), "r") as f:
            content = f.read()
            assert "ThemeProvider" in content, "ThemeProvider not included in layout"
            assert "from '../utils/theme-context'" in content, "Theme context not imported in layout"

    def test_layout_includes_theme_toggle(self):
        """Test that root layout includes ThemeToggle component."""
        with open(get_project_path("frontend/app/layout.tsx"), "r") as f:
            content = f.read()
            assert "ThemeToggle" in content, "ThemeToggle not included in layout"
            assert "from '../components/ThemeToggle'" in content, "ThemeToggle not imported in layout"

    def test_globals_css_has_css_variables(self):
        """Test that globals.css defines CSS variables for theming."""
        with open(get_project_path("frontend/app/globals.css"), "r") as f:
            content = f.read()
            assert ":root" in content, ":root selector not found"
            assert "--color-bg-primary" in content, "Background color variable not defined"
            assert "--color-text-primary" in content, "Text color variable not defined"

    def test_globals_css_has_dark_theme(self):
        """Test that globals.css defines dark theme styles."""
        with open(get_project_path("frontend/app/globals.css"), "r") as f:
            content = f.read()
            assert "[data-theme='dark']" in content or '[data-theme="dark"]' in content, "Dark theme styles not defined"

    def test_globals_css_has_theme_toggle_styles(self):
        """Test that globals.css has styles for theme toggle button."""
        with open(get_project_path("frontend/app/globals.css"), "r") as f:
            content = f.read()
            assert ".theme-toggle" in content, "Theme toggle styles not found"
            assert ".theme-toggle-container" in content, "Theme toggle container styles not found"

    def test_dark_mode_css_variables_comprehensive(self):
        """Test that CSS variables cover all necessary color properties."""
        with open(get_project_path("frontend/app/globals.css"), "r") as f:
            content = f.read()
            required_vars = [
                "--color-bg-primary",
                "--color-bg-secondary",
                "--color-text-primary",
                "--color-text-secondary",
                "--color-border",
                "--color-accent",
            ]
            for var in required_vars:
                assert var in content, f"Required CSS variable not found: {var}"

    def test_theme_context_client_component(self):
        """Test that theme context is marked as a client component."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "'use client'" in content, "Theme context should be a client component"

    def test_theme_toggle_client_component(self):
        """Test that theme toggle is marked as a client component."""
        with open(get_project_path("frontend/components/ThemeToggle.tsx"), "r") as f:
            content = f.read()
            assert "'use client'" in content, "ThemeToggle should be a client component"

    def test_theme_context_has_mounted_state(self):
        """Test that theme context prevents hydration mismatches with mounted state."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "mounted" in content or "Mounted" in content, "Mounted state not found (prevents hydration issues)"

    def test_body_uses_css_variables(self):
        """Test that body element uses CSS variables for theming."""
        with open(get_project_path("frontend/app/globals.css"), "r") as f:
            content = f.read()
            # Check if any body selector uses var() for colors
            assert "body {" in content, "Body selector not found"
            # Look for var(--color- anywhere in the file where body is styled
            assert "var(--color-" in content, "CSS variables should be used somewhere"
            # Specifically check that body uses color and background variables
            assert "color: var(--color-text-primary)" in content or "background: var(--color-bg-primary)" in content, "Body should use CSS variables for theming"

    def test_theme_toggle_has_accessibility(self):
        """Test that theme toggle has proper accessibility attributes."""
        with open(get_project_path("frontend/components/ThemeToggle.tsx"), "r") as f:
            content = f.read()
            assert "aria-label" in content, "Theme toggle should have aria-label"
            assert "title" in content or "aria-label" in content, "Theme toggle should have accessible label"

    def test_css_transition_smooth(self):
        """Test that CSS includes smooth transitions for theme changes."""
        with open(get_project_path("frontend/app/globals.css"), "r") as f:
            content = f.read()
            assert "transition" in content, "CSS should include transitions for smooth theme changes"

    def test_theme_types_defined(self):
        """Test that theme types are properly defined."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "type Theme" in content or "Theme =" in content, "Theme type not defined"
            assert "'light'" in content and "'dark'" in content, "Light and dark theme values not defined"

    def test_theme_context_error_handling(self):
        """Test that useTheme hook has proper error handling."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "throw new Error" in content or "throw Error" in content, "useTheme should throw error when used outside provider"

    def test_theme_toggle_position_fixed(self):
        """Test that theme toggle is positioned fixed for visibility."""
        with open(get_project_path("frontend/app/globals.css"), "r") as f:
            content = f.read()
            toggle_container_section = content[content.find(".theme-toggle-container"):] if ".theme-toggle-container" in content else ""
            assert "position: fixed" in toggle_container_section or "position:fixed" in toggle_container_section, "Theme toggle container should be fixed position"

    def test_theme_context_checks_system_preference(self):
        """Test that theme context checks system color scheme preference."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "prefers-color-scheme" in content, "Should check system color scheme preference"
            assert "matchMedia" in content, "Should use matchMedia to check system preference"

    def test_theme_applies_to_document(self):
        """Test that theme is applied to document root element."""
        with open(get_project_path("frontend/utils/theme-context.tsx"), "r") as f:
            content = f.read()
            assert "document.documentElement" in content or "document.body" in content, "Theme should be applied to document"
            assert "setAttribute" in content or "classList" in content, "Theme should be set via attribute or class"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
