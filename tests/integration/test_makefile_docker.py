"""
Makefile Docker Target Tests

Phase C - Docker Deployment: Tests for Makefile Docker commands.

This test suite validates that the Makefile contains the required Docker
targets with correct commands. Tests use string matching against Makefile
content to verify target definitions without executing them.

Targets verified:
1. docker-build - Build Docker images
2. docker-up - Start containers in detached mode
3. docker-down - Stop and remove containers
4. docker-logs - Follow container logs
5. docker-shell-backend - Open shell in backend container
6. docker-clean - Remove containers, volumes, and images
"""

import re
from pathlib import Path

import pytest


class TestMakefileDockerTargets:
    """Tests for Makefile Docker target definitions."""

    @pytest.fixture
    def makefile_path(self) -> Path:
        """Get the path to the Makefile."""
        return Path(__file__).parent.parent.parent / "Makefile"

    @pytest.fixture
    def makefile_content(self, makefile_path: Path) -> str:
        """Read the Makefile content."""
        assert makefile_path.exists(), f"Makefile not found at {makefile_path}"
        return makefile_path.read_text()

    def test_makefile_exists(self, makefile_path: Path) -> None:
        """Makefile should exist in project root."""
        assert makefile_path.exists(), "Makefile does not exist"

    def test_docker_targets_declared_phony(self, makefile_content: str) -> None:
        """Docker targets should be declared as .PHONY."""
        assert "docker-build" in makefile_content, "docker-build should be in Makefile"
        assert "docker-up" in makefile_content, "docker-up should be in Makefile"
        assert "docker-down" in makefile_content, "docker-down should be in Makefile"
        assert "docker-logs" in makefile_content, "docker-logs should be in Makefile"
        assert "docker-shell-backend" in makefile_content, (
            "docker-shell-backend should be in Makefile"
        )
        assert "docker-clean" in makefile_content, "docker-clean should be in Makefile"
        # Verify they're in .PHONY declaration
        phony_match = re.search(r"\.PHONY:.*", makefile_content, re.DOTALL)
        assert phony_match, "Makefile should have .PHONY declaration"
        phony_section = phony_match.group(0).split("\n")[0:5]  # Check first few lines
        phony_text = " ".join(phony_section)
        assert "docker-build" in phony_text, "docker-build should be .PHONY"

    def test_docker_build_target_exists(self, makefile_content: str) -> None:
        """docker-build target should exist and run docker compose build."""
        assert re.search(r"^docker-build:", makefile_content, re.MULTILINE), (
            "docker-build target should exist"
        )
        # Find the target and its recipe
        match = re.search(r"docker-build:.*?\n((?:\t.*\n)*)", makefile_content)
        assert match, "docker-build target should have recipe"
        recipe = match.group(1)
        assert "docker compose build" in recipe or "docker-compose build" in recipe, (
            "docker-build should run docker compose build"
        )

    def test_docker_up_target_exists(self, makefile_content: str) -> None:
        """docker-up target should exist and run docker compose up -d."""
        assert re.search(r"^docker-up:", makefile_content, re.MULTILINE), (
            "docker-up target should exist"
        )
        # Find the target and its recipe
        match = re.search(r"docker-up:.*?\n((?:\t.*\n)*)", makefile_content)
        assert match, "docker-up target should have recipe"
        recipe = match.group(1)
        assert "docker compose up -d" in recipe or "docker-compose up -d" in recipe, (
            "docker-up should run docker compose up -d"
        )

    def test_docker_down_target_exists(self, makefile_content: str) -> None:
        """docker-down target should exist and run docker compose down."""
        assert re.search(r"^docker-down:", makefile_content, re.MULTILINE), (
            "docker-down target should exist"
        )
        # Find the target and its recipe
        match = re.search(r"docker-down:.*?\n((?:\t.*\n)*)", makefile_content)
        assert match, "docker-down target should have recipe"
        recipe = match.group(1)
        assert "docker compose down" in recipe or "docker-compose down" in recipe, (
            "docker-down should run docker compose down"
        )

    def test_docker_logs_target_exists(self, makefile_content: str) -> None:
        """docker-logs target should exist and run docker compose logs -f."""
        assert re.search(r"^docker-logs:", makefile_content, re.MULTILINE), (
            "docker-logs target should exist"
        )
        # Find the target and its recipe
        match = re.search(r"docker-logs:.*?\n((?:\t.*\n)*)", makefile_content)
        assert match, "docker-logs target should have recipe"
        recipe = match.group(1)
        assert "docker compose logs -f" in recipe or "docker-compose logs -f" in recipe, (
            "docker-logs should run docker compose logs -f"
        )

    def test_docker_shell_backend_target_exists(self, makefile_content: str) -> None:
        """docker-shell-backend target should exist and run docker compose exec backend bash."""
        assert re.search(r"^docker-shell-backend:", makefile_content, re.MULTILINE), (
            "docker-shell-backend target should exist"
        )
        # Find the target and its recipe
        match = re.search(r"docker-shell-backend:.*?\n((?:\t.*\n)*)", makefile_content)
        assert match, "docker-shell-backend target should have recipe"
        recipe = match.group(1)
        exec_cmd = "docker compose exec backend bash"
        exec_cmd_alt = "docker-compose exec backend bash"
        assert exec_cmd in recipe or exec_cmd_alt in recipe, (
            "docker-shell-backend should run docker compose exec backend bash"
        )

    def test_docker_clean_target_exists(self, makefile_content: str) -> None:
        """docker-clean target should exist and run docker compose down -v --rmi all."""
        assert re.search(r"^docker-clean:", makefile_content, re.MULTILINE), (
            "docker-clean target should exist"
        )
        # Find the target and its recipe
        match = re.search(r"docker-clean:.*?\n((?:\t.*\n)*)", makefile_content)
        assert match, "docker-clean target should have recipe"
        recipe = match.group(1)
        assert "-v" in recipe, "docker-clean should remove volumes (-v)"
        assert "--rmi all" in recipe, "docker-clean should remove all images (--rmi all)"

    def test_docker_section_in_help(self, makefile_content: str) -> None:
        """Help target should include Docker commands section."""
        # Look for Docker section in help output
        assert "Docker:" in makefile_content or "docker" in makefile_content.lower(), (
            "Help should include Docker section"
        )
        # Verify help mentions docker-build
        assert re.search(r"make docker-build", makefile_content), (
            "Help should document docker-build command"
        )

    def test_docker_up_shows_urls(self, makefile_content: str) -> None:
        """docker-up target should display URLs after starting containers."""
        match = re.search(r"docker-up:.*?\n((?:\t.*\n)*)", makefile_content)
        assert match, "docker-up target should have recipe"
        recipe = match.group(1)
        # Check for URL display (echo with localhost or port info)
        assert "localhost" in recipe or "Backend:" in recipe or "Frontend:" in recipe, (
            "docker-up should display service URLs"
        )

    def test_docker_targets_use_color_output(self, makefile_content: str) -> None:
        """Docker targets should use colored output for consistency."""
        # Check that docker targets use CYAN/GREEN color variables
        match = re.search(r"docker-build:.*?\n((?:\t.*\n)*)", makefile_content)
        assert match, "docker-build target should have recipe"
        recipe = match.group(1)
        assert "CYAN" in recipe or "GREEN" in recipe, (
            "Docker targets should use colored output"
        )
