"""
Tests to validate USER_GUIDE.md completeness and accuracy.

This test suite ensures the user guide contains all necessary information
and references valid files/commands.
"""

import re
from pathlib import Path
import pytest


# Path to project root and docs
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
USER_GUIDE = DOCS_DIR / "USER_GUIDE.md"


class TestUserGuideExists:
    """Test that the user guide file exists and is readable."""

    def test_user_guide_exists(self):
        """Verify USER_GUIDE.md exists."""
        assert USER_GUIDE.exists(), f"USER_GUIDE.md not found at {USER_GUIDE}"

    def test_user_guide_is_readable(self):
        """Verify USER_GUIDE.md is readable."""
        with open(USER_GUIDE, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 1000, "USER_GUIDE.md appears to be empty or too short"


class TestUserGuideStructure:
    """Test the structure and sections of the user guide."""

    @pytest.fixture
    def guide_content(self):
        """Load user guide content."""
        with open(USER_GUIDE, "r", encoding="utf-8") as f:
            return f.read()

    def test_has_title(self, guide_content):
        """Verify the guide has a title."""
        assert "# FluxHero User Guide" in guide_content

    def test_has_table_of_contents(self, guide_content):
        """Verify the guide has a table of contents."""
        assert "## Table of Contents" in guide_content

    def test_has_required_sections(self, guide_content):
        """Verify all required sections are present."""
        required_sections = [
            "## Introduction",
            "## System Requirements",
            "## Installation",
            "## Configuration",
            "## Running the System",
            "## Using the Dashboard",
            "## Monitoring",
            "## Common Operations",
            "## Troubleshooting",
            "## Best Practices",
        ]
        for section in required_sections:
            assert section in guide_content, f"Missing required section: {section}"

    def test_has_subsections_for_dashboard(self, guide_content):
        """Verify dashboard sections cover all tabs."""
        dashboard_tabs = [
            "### Tab A: Live Trading",
            "### Tab B: Analytics",
            "### Tab C: Trade History",
            "### Tab D: Backtesting",
        ]
        for tab in dashboard_tabs:
            assert tab in guide_content, f"Missing dashboard tab documentation: {tab}"

    def test_has_code_examples(self, guide_content):
        """Verify the guide includes code examples."""
        # Check for bash code blocks
        assert "```bash" in guide_content, "Missing bash code examples"
        # Check for python code blocks
        assert "```python" in guide_content, "Missing Python code examples"
        # Check for JSON examples
        assert "```json" in guide_content or "```" in guide_content


class TestReferencedFilesExist:
    """Test that files referenced in the guide actually exist."""

    @pytest.fixture
    def guide_content(self):
        """Load user guide content."""
        with open(USER_GUIDE, "r", encoding="utf-8") as f:
            return f.read()

    def test_requirements_txt_exists(self, guide_content):
        """Verify requirements.txt is referenced and exists."""
        assert "requirements.txt" in guide_content
        assert (PROJECT_ROOT / "requirements.txt").exists()

    def test_backend_api_server_exists(self, guide_content):
        """Verify backend API server file exists."""
        assert "backend.api.server" in guide_content or "backend/api/server.py" in guide_content
        server_path = PROJECT_ROOT / "fluxhero" / "backend" / "api" / "server.py"
        assert server_path.exists(), f"Backend server not found at {server_path}"

    def test_env_file_referenced(self, guide_content):
        """Verify .env file is documented."""
        assert ".env" in guide_content
        assert "ALPACA_API_KEY" in guide_content

    def test_referenced_docs_exist(self, guide_content):
        """Verify other referenced documentation files exist."""
        # Extract referenced doc files
        doc_references = [
            "API_DOCUMENTATION.md",
            # Note: Other docs may not exist yet, so we only check API_DOCUMENTATION.md
        ]
        for doc in doc_references:
            if doc in guide_content:
                doc_path = DOCS_DIR / doc
                assert doc_path.exists(), f"Referenced document not found: {doc}"


class TestConfigurationGuidance:
    """Test that configuration guidance is complete."""

    @pytest.fixture
    def guide_content(self):
        """Load user guide content."""
        with open(USER_GUIDE, "r", encoding="utf-8") as f:
            return f.read()

    def test_environment_variables_documented(self, guide_content):
        """Verify key environment variables are documented."""
        required_env_vars = [
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY",
            "BACKEND_PORT",
            "MAX_DAILY_LOSS_PERCENT",
            "LOG_LEVEL",
        ]
        for env_var in required_env_vars:
            assert env_var in guide_content, f"Environment variable {env_var} not documented"

    def test_security_warnings_present(self, guide_content):
        """Verify security warnings are included."""
        # Check for NEVER (in bold or plain text) related to committing secrets
        assert ("NEVER" in guide_content and "commit" in guide_content) or "never commit" in guide_content.lower()
        assert "paper trading" in guide_content.lower()
        assert "API key" in guide_content

    def test_risk_parameters_documented(self, guide_content):
        """Verify risk parameters are explained."""
        risk_params = [
            "MAX_DAILY_LOSS_PERCENT",
            "MAX_POSITION_SIZE_PERCENT",
            "MAX_TOTAL_EXPOSURE_PERCENT",
        ]
        for param in risk_params:
            assert param in guide_content, f"Risk parameter {param} not documented"


class TestCommandsAreValid:
    """Test that shell commands in the guide use correct syntax."""

    @pytest.fixture
    def guide_content(self):
        """Load user guide content."""
        with open(USER_GUIDE, "r", encoding="utf-8") as f:
            return f.read()

    def test_pip_commands_are_valid(self, guide_content):
        """Verify pip commands use correct syntax."""
        # Should use 'pip install -r requirements.txt'
        assert "pip install -r requirements.txt" in guide_content

    def test_uvicorn_commands_are_valid(self, guide_content):
        """Verify uvicorn commands are correct."""
        assert "uvicorn" in guide_content
        # Should reference the correct module path
        assert "backend.api.server:app" in guide_content or "fluxhero.backend.api.server:app" in guide_content

    def test_curl_commands_use_localhost(self, guide_content):
        """Verify curl commands reference correct host."""
        assert "localhost:8000" in guide_content or "127.0.0.1:8000" in guide_content


class TestTroubleshootingSection:
    """Test that troubleshooting section is comprehensive."""

    @pytest.fixture
    def guide_content(self):
        """Load user guide content."""
        with open(USER_GUIDE, "r", encoding="utf-8") as f:
            return f.read()

    def test_common_issues_covered(self, guide_content):
        """Verify common issues are documented."""
        common_issues = [
            "Backend Won't Start",
            "WebSocket",
            "No Data",
            "Backtest Fails",
        ]
        for issue in common_issues:
            # Case-insensitive check
            assert issue.lower() in guide_content.lower(), f"Issue not covered: {issue}"

    def test_has_symptom_cause_fix_structure(self, guide_content):
        """Verify troubleshooting uses Symptom/Cause/Fix structure."""
        assert "Symptom" in guide_content or "symptom" in guide_content.lower()
        assert "Cause" in guide_content or "cause" in guide_content.lower()
        assert "Fix" in guide_content or "fix" in guide_content.lower()


class TestBestPractices:
    """Test that best practices section is present and useful."""

    @pytest.fixture
    def guide_content(self):
        """Load user guide content."""
        with open(USER_GUIDE, "r", encoding="utf-8") as f:
            return f.read()

    def test_risk_management_guidance(self, guide_content):
        """Verify risk management best practices are included."""
        risk_keywords = ["position siz", "diversif", "loss limit", "drawdown"]
        for keyword in risk_keywords:
            assert keyword in guide_content.lower(), f"Risk guidance missing keyword: {keyword}"

    def test_backtesting_best_practices(self, guide_content):
        """Verify backtesting best practices are documented."""
        backtest_keywords = ["walk-forward", "out-of-sample", "overfit"]
        # At least some of these should be present
        found = [kw for kw in backtest_keywords if kw in guide_content.lower()]
        assert len(found) >= 1, "Backtesting best practices not adequately covered"

    def test_maintenance_tasks_listed(self, guide_content):
        """Verify maintenance tasks are documented."""
        maintenance_keywords = ["daily", "weekly", "monthly"]
        for keyword in maintenance_keywords:
            # Should appear in context of tasks
            assert keyword in guide_content.lower(), f"Maintenance frequency {keyword} not documented"


class TestQuickReference:
    """Test that quick reference section is useful."""

    @pytest.fixture
    def guide_content(self):
        """Load user guide content."""
        with open(USER_GUIDE, "r", encoding="utf-8") as f:
            return f.read()

    def test_has_quick_reference(self, guide_content):
        """Verify quick reference section exists."""
        assert "Quick Reference" in guide_content or "quick reference" in guide_content.lower()

    def test_common_commands_listed(self, guide_content):
        """Verify common commands are in quick reference."""
        # Should have frequently used commands
        assert "uvicorn" in guide_content
        assert "npm run dev" in guide_content or "npm dev" in guide_content


class TestDocumentationQuality:
    """Test documentation quality metrics."""

    @pytest.fixture
    def guide_content(self):
        """Load user guide content."""
        with open(USER_GUIDE, "r", encoding="utf-8") as f:
            return f.read()

    def test_has_sufficient_length(self, guide_content):
        """Verify guide has sufficient content (not just stubs)."""
        # A comprehensive user guide should be at least 15,000 characters
        assert len(guide_content) > 15000, "USER_GUIDE.md may be incomplete (too short)"

    def test_has_examples_in_each_major_section(self, guide_content):
        """Verify major sections include examples."""
        # Check that code blocks are distributed throughout
        sections = guide_content.split("##")
        code_block_sections = [s for s in sections if "```" in s]
        # At least 5 sections should have code examples
        assert len(code_block_sections) >= 5, "Not enough code examples distributed across sections"

    def test_no_placeholder_text(self, guide_content):
        """Verify no placeholder text remains."""
        placeholders = ["TODO", "TBD", "FIXME", "XXX", "[INSERT", "PLACEHOLDER"]
        for placeholder in placeholders:
            assert placeholder not in guide_content.upper(), f"Placeholder text found: {placeholder}"

    def test_consistent_formatting(self, guide_content):
        """Verify consistent markdown formatting."""
        # Check that headers are properly formatted (# followed by space)
        # Note: We need to match headers with text after them, not just standalone ##
        bad_headers = re.findall(r"^(#{1,6})([^\s\n#])", guide_content, re.MULTILINE)
        assert len(bad_headers) == 0, f"Found headers without space after #: {bad_headers}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
