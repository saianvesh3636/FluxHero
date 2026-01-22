"""
Unit tests to verify FluxHero project folder structure.

Tests that all required directories and __init__.py files exist.
"""

import os
from pathlib import Path

# Get project root directory (3 levels up from this test file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestProjectStructure:
    """Test suite for verifying project folder structure."""

    def test_project_root_exists(self):
        """Verify project root directory exists."""
        assert PROJECT_ROOT.is_dir()

    def test_backend_directories_exist(self):
        """Verify all backend module directories exist."""
        backend_modules = [
            "computation",
            "strategy",
            "storage",
            "data",
            "backtesting",
            "execution",
            "risk",
            "api",
        ]

        for module in backend_modules:
            module_path = PROJECT_ROOT / "backend" / module
            assert module_path.is_dir(), f"Backend module '{module}' directory not found"

    def test_backend_init_files_exist(self):
        """Verify __init__.py files exist for all backend modules."""
        backend_modules = [
            "backend",  # backend/__init__.py
            "backend/computation",
            "backend/strategy",
            "backend/storage",
            "backend/data",
            "backend/backtesting",
            "backend/execution",
            "backend/risk",
            "backend/api",
        ]

        for module in backend_modules:
            init_file = PROJECT_ROOT / module / "__init__.py"
            assert init_file.is_file(), f"__init__.py not found in {module}"

    def test_frontend_directories_exist(self):
        """Verify all frontend module directories exist."""
        frontend_modules = [
            "pages",
            "components",
            "utils",
            "styles",
        ]

        for module in frontend_modules:
            module_path = PROJECT_ROOT / "frontend" / module
            assert module_path.is_dir(), f"Frontend module '{module}' directory not found"

    def test_test_directories_exist(self):
        """Verify test directory structure exists."""
        test_dirs = [
            "unit",
            "integration",
            "e2e",
        ]

        for test_dir in test_dirs:
            dir_path = PROJECT_ROOT / "tests" / test_dir
            assert dir_path.is_dir(), f"Test directory '{test_dir}' not found"

    def test_test_init_files_exist(self):
        """Verify __init__.py files exist for test directories."""
        test_dirs = [
            "",  # tests/__init__.py
            "unit",
            "integration",
            "e2e",
        ]

        for test_dir in test_dirs:
            init_file = PROJECT_ROOT / "tests" / test_dir / "__init__.py"
            assert init_file.is_file(), f"__init__.py not found in tests/{test_dir}"

    def test_data_directories_exist(self):
        """Verify data storage directories exist."""
        data_dirs = [
            "cache",
            "archive",
        ]

        for data_dir in data_dirs:
            dir_path = PROJECT_ROOT / "data" / data_dir
            assert dir_path.is_dir(), f"Data directory '{data_dir}' not found"

    def test_logs_directory_exists(self):
        """Verify logs directory exists."""
        assert (PROJECT_ROOT / "logs").is_dir()

    def test_config_directory_exists(self):
        """Verify config directory exists."""
        assert (PROJECT_ROOT / "config").is_dir()

    def test_readme_exists(self):
        """Verify README documentation exists."""
        readme_path = PROJECT_ROOT / "README.md"
        assert readme_path.is_file(), "README.md not found"

        # Verify README has content
        content = readme_path.read_text()
        assert len(content) > 0, "README.md is empty"

    def test_python_packages_are_importable(self):
        """Verify Python packages can be imported (have __init__.py)."""
        # This test verifies the packages are structured correctly
        # Actual imports would require the modules to be in sys.path

        required_packages = [
            "backend",
            "backend/computation",
            "backend/strategy",
            "backend/storage",
            "backend/data",
            "backend/backtesting",
            "backend/execution",
            "backend/risk",
            "backend/api",
        ]

        for package in required_packages:
            package_path = PROJECT_ROOT / package.replace("/", os.sep)
            init_file = package_path / "__init__.py"
            assert package_path.is_dir(), f"Package directory {package} not found"
            assert init_file.is_file(), f"Package {package} missing __init__.py"
