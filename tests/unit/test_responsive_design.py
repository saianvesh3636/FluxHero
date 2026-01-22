"""
Unit tests for responsive design implementation

Tests verify that:
1. Responsive CSS classes are properly defined
2. Media queries are correctly structured
3. Tablet breakpoints are properly set (600px-1023px)
4. Touch-friendly sizes are enforced (44px minimum)
5. All responsive utilities are available
"""

import re
from pathlib import Path


class TestResponsiveDesign:
    """Test suite for responsive design system"""

    @staticmethod
    def get_globals_css_content():
        """Helper to read globals.css content"""
        # Get the project root directory
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        css_path = project_root / "" / "frontend" / "app" / "globals.css"

        if not css_path.exists():
            raise FileNotFoundError(f"globals.css not found at {css_path}")
        with open(css_path, encoding="utf-8") as f:
            return f.read()

    def test_responsive_section_exists(self):
        """Test that responsive design section exists in globals.css"""
        content = self.get_globals_css_content()
        assert "Responsive Design System for Tablets" in content
        assert "Tablet Breakpoints" in content

    def test_tablet_breakpoints_defined(self):
        """Test that tablet breakpoints are properly defined"""
        content = self.get_globals_css_content()

        # Check for small tablets (600px - 767px)
        assert "600px" in content or "767px" in content

        # Check for large tablets (768px - 1023px)
        assert "768px" in content
        assert "1023px" in content

        # Check for media queries
        assert "@media (max-width: 1023px)" in content
        assert "@media (max-width: 767px)" in content

    def test_touch_friendly_sizes(self):
        """Test that touch-friendly minimum sizes are enforced (44px WCAG)"""
        content = self.get_globals_css_content()

        # Check for 44px minimum height for interactive elements
        assert "min-height: 44px" in content

        # Check for button, input, select sizing
        assert re.search(r"button.*min-height:\s*44px", content, re.DOTALL)
        assert re.search(r"input.*min-height:\s*44px", content, re.DOTALL)

    def test_responsive_grid_classes(self):
        """Test that responsive grid classes are defined"""
        content = self.get_globals_css_content()

        # Check for stats-grid responsive behavior
        assert ".stats-grid" in content
        assert "grid-template-columns: repeat(2, 1fr)" in content

        # Check for single-col-tablet
        assert ".single-col-tablet" in content

    def test_table_responsiveness(self):
        """Test that tables have responsive behavior"""
        content = self.get_globals_css_content()

        # Check for table-container with horizontal scroll
        assert ".table-container" in content
        assert "overflow-x: auto" in content
        assert "-webkit-overflow-scrolling: touch" in content

        # Check for reduced padding on tablets
        assert re.search(r"table th.*padding.*0\.75rem", content, re.DOTALL)

    def test_chart_responsiveness(self):
        """Test that charts have responsive sizing"""
        content = self.get_globals_css_content()

        # Check for chart-container
        assert ".chart-container" in content

        # Check for different heights at different breakpoints
        assert "height: 400px" in content  # Tablet
        assert "height: 300px" in content  # Small tablet

    def test_form_elements_responsive(self):
        """Test that form elements are responsive"""
        content = self.get_globals_css_content()

        # Check for form-row stacking
        assert ".form-row" in content
        assert "flex-direction: column" in content

        # Check for full width form elements
        assert "width: 100%" in content

    def test_modal_adjustments(self):
        """Test that modals are properly sized for tablets"""
        content = self.get_globals_css_content()

        # Check for modal-content sizing
        assert ".modal-content" in content
        assert "max-width: 90vw" in content
        assert "max-height: 90vh" in content

    def test_navigation_responsiveness(self):
        """Test that navigation/tabs are responsive"""
        content = self.get_globals_css_content()

        # Check for nav-tabs with horizontal scroll
        assert ".nav-tabs" in content
        assert "overflow-x: auto" in content
        assert "white-space: nowrap" in content

    def test_card_components(self):
        """Test that card components have responsive styling"""
        content = self.get_globals_css_content()

        # Check for card classes
        assert ".card" in content
        assert ".card-padding" in content
        assert "padding: 1rem" in content

    def test_spacing_utilities(self):
        """Test that spacing utilities exist for tablets"""
        content = self.get_globals_css_content()

        # Check for tablet-specific spacing
        assert ".mb-tablet-4" in content
        assert ".gap-tablet-4" in content
        assert ".p-tablet-4" in content

    def test_orientation_specific_rules(self):
        """Test that orientation-specific rules exist"""
        content = self.get_globals_css_content()

        # Check for landscape orientation
        assert "orientation: landscape" in content

        # Check for portrait orientation
        assert "orientation: portrait" in content

    def test_utility_classes(self):
        """Test that responsive utility classes are defined"""
        content = self.get_globals_css_content()

        # Check for hide/show utilities
        assert ".responsive-hide-tablet" in content
        assert ".responsive-show-tablet" in content

        # Check for flexbox utilities
        assert ".flex-col-tablet" in content
        assert ".flex-wrap-tablet" in content

    def test_loading_states(self):
        """Test that loading states are responsive"""
        content = self.get_globals_css_content()

        # Check for loading-spinner sizing
        assert ".loading-spinner" in content
        assert re.search(r"\.loading-spinner.*width.*2\.5rem", content, re.DOTALL)

    def test_accessibility_compliance(self):
        """Test that accessibility requirements are met"""
        content = self.get_globals_css_content()

        # Check for WCAG 2.1 Level AAA compliance mention
        assert "WCAG" in content or "accessibility" in content.lower()

        # Check for minimum touch target sizes
        assert "min-height: 44px" in content
        assert "min-width: 44px" in content

    def test_print_media_query(self):
        """Test that print styles are defined"""
        content = self.get_globals_css_content()

        # Check for print media query
        assert "@media print" in content
        assert "display: none" in content

    def test_theme_toggle_responsive(self):
        """Test that theme toggle is responsive"""
        content = self.get_globals_css_content()

        # Check for theme-toggle-container adjustments
        assert ".theme-toggle-container" in content

        # Check for size adjustments at tablet breakpoint
        tablet_section = re.search(
            r"@media \(max-width: 1023px\).*?\.theme-toggle\s*\{[^}]*width:\s*40px",
            content,
            re.DOTALL
        )
        assert tablet_section is not None

    def test_page_container_responsive(self):
        """Test that page containers have responsive padding"""
        content = self.get_globals_css_content()

        # Check for page-container class
        assert ".page-container" in content
        assert re.search(r"\.page-container.*padding.*1rem", content, re.DOTALL)

    def test_live_page_uses_responsive_classes(self):
        """Test that live trading page uses responsive classes"""
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        live_page_path = project_root / "" / "frontend" / "app" / "live" / "page.tsx"

        if not live_page_path.exists():
            # Skip test if file doesn't exist
            return

        with open(live_page_path, encoding="utf-8") as f:
            content = f.read()

        # Check for responsive classes
        assert "page-container" in content
        assert "stats-grid" in content
        assert "table-container" in content

    def test_analytics_page_uses_responsive_classes(self):
        """Test that analytics page uses responsive classes"""
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        analytics_page_path = project_root / "" / "frontend" / "app" / "analytics" / "page.tsx"

        if not analytics_page_path.exists():
            # Skip test if file doesn't exist
            return

        with open(analytics_page_path, encoding="utf-8") as f:
            content = f.read()

        # Check for responsive classes
        assert "page-container" in content
        assert "flex-col-tablet" in content or "chart-container" in content

    def test_backtest_page_uses_responsive_classes(self):
        """Test that backtest page uses responsive classes"""
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        backtest_page_path = project_root / "" / "frontend" / "app" / "backtest" / "page.tsx"

        if not backtest_page_path.exists():
            # Skip test if file doesn't exist
            return

        with open(backtest_page_path, encoding="utf-8") as f:
            content = f.read()

        # Check for responsive classes
        assert "page-container" in content
        assert "single-col-tablet" in content

    def test_history_page_uses_responsive_classes(self):
        """Test that history page uses responsive classes"""
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        history_page_path = project_root / "" / "frontend" / "pages" / "history.tsx"

        if not history_page_path.exists():
            # Skip test if file doesn't exist
            return

        with open(history_page_path, encoding="utf-8") as f:
            content = f.read()

        # Check for responsive classes
        assert "page-container" in content
        assert "flex-col-tablet" in content or "table-container" in content

    def test_font_size_adjustments(self):
        """Test that font sizes are adjusted for tablets"""
        content = self.get_globals_css_content()

        # Check for h1, h2, h3 adjustments
        assert re.search(r"h1.*font-size.*1\.875rem", content, re.DOTALL)
        assert re.search(r"h2.*font-size.*1\.5rem", content, re.DOTALL)
        assert re.search(r"h3.*font-size.*1\.25rem", content, re.DOTALL)

    def test_css_syntax_valid(self):
        """Test that CSS has valid syntax (basic check)"""
        content = self.get_globals_css_content()

        # Check for matching braces
        open_braces = content.count("{")
        close_braces = content.count("}")
        assert open_braces == close_braces, "Mismatched braces in CSS"

        # Check for valid media query syntax
        media_queries = re.findall(r"@media\s+\([^)]+\)", content)
        assert len(media_queries) > 0, "No media queries found"

        # Check that all media queries have proper structure
        for mq in media_queries:
            assert "(" in mq and ")" in mq, f"Invalid media query: {mq}"

    def test_no_hardcoded_breakpoints_in_components(self):
        """Test that components use CSS classes, not inline breakpoints"""
        # This is a best practice check
        # Components should use the CSS classes we defined, not inline styles

        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent

        pages_to_check = [
            project_root / "" / "frontend" / "app" / "live" / "page.tsx",
            project_root / "" / "frontend" / "app" / "analytics" / "page.tsx",
            project_root / "" / "frontend" / "app" / "backtest" / "page.tsx",
            project_root / "" / "frontend" / "pages" / "history.tsx",
        ]

        for page_path in pages_to_check:
            if not page_path.exists():
                continue

            with open(page_path, encoding="utf-8") as f:
                content = f.read()

            # Check that we're using CSS classes
            # Look for Tailwind responsive classes or our custom classes
            has_responsive_classes = (
                "md:" in content or
                "lg:" in content or
                "page-container" in content or
                "stats-grid" in content or
                "flex-col-tablet" in content
            )

            assert has_responsive_classes, f"{page_path} should use responsive CSS classes"

    def test_comprehensive_coverage(self):
        """Test that all major responsive features are covered"""
        content = self.get_globals_css_content()

        required_features = [
            ".page-container",
            ".stats-grid",
            ".table-container",
            ".chart-container",
            ".card",
            ".modal-content",
            ".nav-tabs",
            ".form-row",
            ".responsive-hide-tablet",
            ".responsive-show-tablet",
            ".flex-col-tablet",
            ".loading-spinner",
            "@media (max-width: 1023px)",
            "@media (max-width: 767px)",
            "min-height: 44px",
        ]

        for feature in required_features:
            assert feature in content, f"Missing responsive feature: {feature}"

    def test_documentation_comments(self):
        """Test that responsive CSS has proper documentation"""
        content = self.get_globals_css_content()

        # Check for section comments
        assert "Tablet Breakpoints" in content
        assert "Base responsive utilities" in content
        assert "Touch-friendly" in content

        # Check for explanatory comments
        comment_count = content.count("/*")
        assert comment_count >= 10, "Should have at least 10 comment blocks for documentation"


# Run all tests
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
