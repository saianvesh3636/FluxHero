"""
Unit tests for validating requirements.txt file.
"""

from pathlib import Path


def test_requirements_file_exists():
    """Test that requirements.txt exists in project root."""
    requirements_path = Path("requirements.txt")
    assert requirements_path.exists(), "requirements.txt should exist in project root"


def test_venv_directory_exists():
    """Test that virtual environment directory exists."""
    venv_path = Path(".venv")
    assert venv_path.exists(), "venv directory should exist in project root"
    assert venv_path.is_dir(), "venv should be a directory"


def test_requirements_contains_core_dependencies():
    """Test that requirements.txt contains all required core dependencies."""
    requirements_path = Path("requirements.txt")

    with open(requirements_path, "r") as f:
        content = f.read()

    # Required core dependencies from the task
    required_deps = [
        "numba",
        "numpy",
        "pandas",
        "httpx",
        "websockets",
        "fastapi",
        "pyarrow",
    ]

    for dep in required_deps:
        assert dep in content, f"{dep} should be in requirements.txt"


def test_requirements_contains_testing_dependencies():
    """Test that requirements.txt contains testing dependencies."""
    requirements_path = Path("requirements.txt")

    with open(requirements_path, "r") as f:
        content = f.read()

    # Testing dependencies
    testing_deps = [
        "pytest",
    ]

    for dep in testing_deps:
        assert dep in content, f"{dep} should be in requirements.txt"


def test_requirements_valid_format():
    """Test that requirements.txt has valid format (no empty package names)."""
    requirements_path = Path("requirements.txt")

    with open(requirements_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue

        # Check that line contains a package name
        assert len(line) > 0, "Package line should not be empty"

        # Check that it doesn't start with special characters (except comments)
        if not line.startswith("#"):
            assert line[0].isalnum(), f"Package name should start with alphanumeric character: {line}"


def test_requirements_contains_version_specs():
    """Test that core dependencies have version specifications."""
    requirements_path = Path("requirements.txt")

    with open(requirements_path, "r") as f:
        content = f.read()

    # Core dependencies that should have version specs
    core_deps_with_versions = [
        ("numba", ">="),
        ("numpy", ">="),
        ("pandas", ">="),
        ("httpx", ">="),
        ("websockets", ">="),
        ("fastapi", ">="),
        ("pyarrow", ">="),
    ]

    for dep, operator in core_deps_with_versions:
        # Check that the dependency appears with a version operator
        found_with_version = False
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith(dep) and operator in line:
                found_with_version = True
                break

        assert found_with_version, f"{dep} should have a version specification with {operator}"


def test_sqlite3_note_present():
    """Test that requirements.txt mentions sqlite3 is in standard library."""
    requirements_path = Path("requirements.txt")

    with open(requirements_path, "r") as f:
        content = f.read()

    # Should mention sqlite3 in comments
    assert "sqlite3" in content.lower(), "requirements.txt should note that sqlite3 is in standard library"
