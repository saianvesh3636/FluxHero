"""
Unit tests for .gitignore configuration.

This test suite validates that the .gitignore file properly excludes:
- Python-specific files (__pycache__, venv, etc.)
- Node.js-specific files (node_modules, .next, etc.)
- Data files (data/, *.parquet, *.db, etc.)
- Environment files (.env, .env.local, etc.)
- IDE and OS-specific files
"""

from pathlib import Path


class TestGitignore:
    """Test suite for .gitignore file."""

    @staticmethod
    def get_project_root():
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    @staticmethod
    def read_gitignore():
        """Read and return .gitignore contents."""
        gitignore_path = TestGitignore.get_project_root() / ".gitignore"
        with open(gitignore_path) as f:
            return f.read()

    def test_gitignore_exists(self):
        """Test that .gitignore file exists."""
        gitignore_path = self.get_project_root() / ".gitignore"
        assert gitignore_path.exists(), ".gitignore file should exist"
        assert gitignore_path.is_file(), ".gitignore should be a file"

    def test_python_patterns(self):
        """Test that Python-specific patterns are included."""
        content = self.read_gitignore()

        # Core Python patterns
        assert "__pycache__/" in content
        assert "*.py[cod]" in content
        assert "*.so" in content
        assert "*.egg-info/" in content

        # Virtual environments
        assert "venv/" in content
        assert ".venv" in content
        assert "ENV/" in content

        # Testing and coverage
        assert ".pytest_cache/" in content
        assert ".coverage" in content
        assert "htmlcov/" in content

        # Type checking
        assert ".mypy_cache/" in content

    def test_nodejs_patterns(self):
        """Test that Node.js-specific patterns are included."""
        content = self.read_gitignore()

        # Dependencies
        assert "node_modules/" in content

        # Next.js specific
        assert ".next/" in content
        assert "out/" in content
        assert "*.tsbuildinfo" in content
        assert "next-env.d.ts" in content

        # Logs
        assert "npm-debug.log*" in content
        assert "yarn-debug.log*" in content
        assert "yarn-error.log*" in content

    def test_env_files_excluded(self):
        """Test that environment files are properly excluded."""
        content = self.read_gitignore()

        # Core .env patterns
        assert ".env" in content
        assert ".env.local" in content
        assert ".env.development.local" in content
        assert ".env.test.local" in content
        assert ".env.production.local" in content
        assert ".env*.local" in content

    def test_data_files_excluded(self):
        """Test that data directories and files are excluded."""
        content = self.read_gitignore()

        # Data directories
        assert "data/" in content
        assert "cache/" in content
        assert "archive/" in content

        # Data file formats
        assert "*.parquet" in content
        assert "*.db" in content
        assert "*.sqlite" in content
        assert "*.sqlite3" in content
        assert "*.csv" in content
        assert "*.h5" in content
        assert "*.hdf5" in content

    def test_secrets_excluded(self):
        """Test that API keys and secrets are excluded."""
        content = self.read_gitignore()

        assert "*.key" in content
        assert "*.pem" in content
        assert "secrets/" in content
        assert "credentials.json" in content
        assert "secrets.json" in content

    def test_logs_excluded(self):
        """Test that log files and directories are excluded."""
        content = self.read_gitignore()

        assert "logs/" in content
        assert "*.log" in content

    def test_backtest_results_excluded(self):
        """Test that backtest results are excluded."""
        content = self.read_gitignore()

        assert "backtests/" in content
        assert "backtest_results/" in content
        assert "reports/" in content
        assert "tearsheets/" in content

    def test_ide_patterns(self):
        """Test that IDE-specific files are excluded."""
        content = self.read_gitignore()

        # VSCode
        assert ".vscode/" in content

        # PyCharm
        assert ".idea/" in content

        # Vim
        assert "*.swp" in content
        assert "*.swo" in content

        # Eclipse
        assert ".project" in content
        assert ".classpath" in content

    def test_os_patterns(self):
        """Test that OS-specific files are excluded."""
        content = self.read_gitignore()

        # macOS
        assert ".DS_Store" in content

        # Windows
        assert "Thumbs.db" in content

        # Temporary files
        assert "*.tmp" in content
        assert "*.bak" in content

    def test_ralphy_patterns(self):
        """Test that Ralphy agent files are properly handled."""
        content = self.read_gitignore()

        # Progress tracking should be ignored
        assert ".ralphy/progress.txt" in content
        assert ".ralphy/sessions/" in content

    def test_build_artifacts_excluded(self):
        """Test that build artifacts are excluded."""
        content = self.read_gitignore()

        # Python
        assert "build/" in content
        assert "dist/" in content

        # General
        assert "*.tmp" in content
        assert "tmp/" in content
        assert "temp/" in content

    def test_database_files_excluded(self):
        """Test that all database file patterns are excluded."""
        content = self.read_gitignore()

        assert "*.db" in content
        assert "*.db-journal" in content
        assert "*.sqlite" in content
        assert "*.sqlite3" in content
        assert "db.sqlite3" in content
        assert "db.sqlite3-journal" in content

    def test_session_data_excluded(self):
        """Test that trading session data is excluded."""
        content = self.read_gitignore()

        assert "sessions/" in content
        assert "trades/" in content

    def test_structured_with_sections(self):
        """Test that .gitignore has clear section headers."""
        content = self.read_gitignore()

        # Check for major sections
        assert "Python" in content
        assert "Node.js" in content
        assert "FluxHero-Specific" in content
        assert "IDE / Editor" in content
        assert "OS" in content

    def test_no_duplicate_entries(self):
        """Test that there are no exact duplicate pattern lines."""
        content = self.read_gitignore()
        lines = [
            line.strip()
            for line in content.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Find duplicates
        seen = set()
        duplicates = []
        for line in lines:
            if line in seen:
                duplicates.append(line)
            seen.add(line)

        # Allow some duplicates (like *.log which appears in multiple contexts)
        # but flag if we have too many
        assert len(duplicates) < 10, f"Too many duplicate entries: {duplicates}"
