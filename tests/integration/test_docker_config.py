"""
Docker Configuration Tests

Phase C - Docker Deployment: Tests for Dockerfile validation and structure.

This test suite covers:
1. Dockerfile.backend existence and structure
   - Uses correct base image
   - Installs uv package manager
   - Sets up virtual environment correctly
   - Exposes correct port
   - Has HEALTHCHECK configured
   - Has correct CMD for uvicorn

2. Dockerfile.frontend existence and structure
   - Uses Node 20 Alpine base
   - Multi-stage build
   - Has HEALTHCHECK configured
   - Uses non-root user

3. docker-compose.yml validation
   - Defines backend and frontend services
   - Correct port mappings
   - Volume mounts for data persistence
   - Health checks configured
   - Service dependencies

4. .dockerignore validation
   - Excludes development files
   - Excludes test files
   - Excludes IDE configurations
   - Excludes __pycache__

5. Docker build prerequisites
   - pyproject.toml exists
   - uv.lock exists
   - backend directory structure is valid
"""

import re
from pathlib import Path

import pytest


class TestBackendDockerfile:
    """Tests for backend Dockerfile structure and content."""

    @pytest.fixture
    def dockerfile_path(self) -> Path:
        """Get the path to the backend Dockerfile."""
        return Path(__file__).parent.parent.parent / "docker" / "Dockerfile.backend"

    @pytest.fixture
    def dockerfile_content(self, dockerfile_path: Path) -> str:
        """Read the Dockerfile content."""
        assert dockerfile_path.exists(), f"Dockerfile not found at {dockerfile_path}"
        return dockerfile_path.read_text()

    def test_dockerfile_exists(self, dockerfile_path: Path) -> None:
        """Dockerfile.backend should exist in docker/ directory."""
        assert dockerfile_path.exists(), "docker/Dockerfile.backend does not exist"

    def test_uses_python_311_slim_base(self, dockerfile_content: str) -> None:
        """Should use python:3.11-slim as base image."""
        assert "FROM python:3.11-slim" in dockerfile_content, (
            "Dockerfile should use python:3.11-slim base image"
        )

    def test_installs_uv_package_manager(self, dockerfile_content: str) -> None:
        """Should install uv package manager via curl."""
        assert "astral.sh/uv/install.sh" in dockerfile_content, (
            "Dockerfile should install uv package manager"
        )

    def test_copies_dependency_files(self, dockerfile_content: str) -> None:
        """Should copy pyproject.toml and uv.lock for dependency installation."""
        assert "COPY pyproject.toml" in dockerfile_content, "Dockerfile should copy pyproject.toml"
        assert "uv.lock" in dockerfile_content, "Dockerfile should reference uv.lock"

    def test_runs_uv_sync(self, dockerfile_content: str) -> None:
        """Should run uv sync for dependency installation."""
        assert "uv sync" in dockerfile_content, "Dockerfile should run uv sync"

    def test_copies_backend_code(self, dockerfile_content: str) -> None:
        """Should copy the backend directory."""
        assert re.search(r"COPY backend[/\s]", dockerfile_content), (
            "Dockerfile should copy backend/ directory"
        )

    def test_exposes_port_8000(self, dockerfile_content: str) -> None:
        """Should expose port 8000 for FastAPI."""
        assert "EXPOSE 8000" in dockerfile_content, "Dockerfile should expose port 8000"

    def test_has_healthcheck(self, dockerfile_content: str) -> None:
        """Should have HEALTHCHECK instruction using /health endpoint."""
        assert "HEALTHCHECK" in dockerfile_content, "Dockerfile should have HEALTHCHECK instruction"
        assert "/health" in dockerfile_content, "HEALTHCHECK should use /health endpoint"

    def test_cmd_runs_uvicorn(self, dockerfile_content: str) -> None:
        """Should have CMD that runs uvicorn with correct app path."""
        assert "uvicorn" in dockerfile_content, "Dockerfile should run uvicorn"
        assert "backend.api.server:app" in dockerfile_content, (
            "Dockerfile should reference backend.api.server:app"
        )

    def test_sets_pythonpath(self, dockerfile_content: str) -> None:
        """Should set PYTHONPATH environment variable."""
        assert "PYTHONPATH" in dockerfile_content, "Dockerfile should set PYTHONPATH"

    def test_sets_pythonunbuffered(self, dockerfile_content: str) -> None:
        """Should set PYTHONUNBUFFERED for proper logging."""
        assert "PYTHONUNBUFFERED" in dockerfile_content, "Dockerfile should set PYTHONUNBUFFERED=1"

    def test_creates_data_directory(self, dockerfile_content: str) -> None:
        """Should create /app/data directory for SQLite and cache."""
        assert "/app/data" in dockerfile_content, "Dockerfile should create /app/data directory"

    def test_creates_logs_directory(self, dockerfile_content: str) -> None:
        """Should create /app/logs directory for log files."""
        assert "/app/logs" in dockerfile_content, "Dockerfile should create /app/logs directory"

    def test_uses_multistage_build(self, dockerfile_content: str) -> None:
        """Should use multi-stage build for smaller image size."""
        from_count = dockerfile_content.count("FROM python:3.11-slim")
        assert from_count >= 2, (
            "Dockerfile should use multi-stage build (at least 2 FROM instructions)"
        )


class TestDockerignore:
    """Tests for .dockerignore file content."""

    @pytest.fixture
    def dockerignore_path(self) -> Path:
        """Get the path to .dockerignore."""
        return Path(__file__).parent.parent.parent / ".dockerignore"

    @pytest.fixture
    def dockerignore_content(self, dockerignore_path: Path) -> str:
        """Read the .dockerignore content."""
        assert dockerignore_path.exists(), ".dockerignore not found"
        return dockerignore_path.read_text()

    def test_dockerignore_exists(self, dockerignore_path: Path) -> None:
        """.dockerignore should exist in project root."""
        assert dockerignore_path.exists(), ".dockerignore does not exist"

    def test_excludes_git_directory(self, dockerignore_content: str) -> None:
        """Should exclude .git directory."""
        assert ".git" in dockerignore_content, ".dockerignore should exclude .git/"

    def test_excludes_pycache(self, dockerignore_content: str) -> None:
        """Should exclude __pycache__ directories."""
        assert "__pycache__" in dockerignore_content, ".dockerignore should exclude __pycache__"

    def test_excludes_node_modules(self, dockerignore_content: str) -> None:
        """Should exclude node_modules directory."""
        assert "node_modules" in dockerignore_content, ".dockerignore should exclude node_modules/"

    def test_excludes_tests(self, dockerignore_content: str) -> None:
        """Should exclude tests directory."""
        assert "tests" in dockerignore_content, ".dockerignore should exclude tests/"

    def test_excludes_ide_configs(self, dockerignore_content: str) -> None:
        """Should exclude IDE configuration directories."""
        assert ".vscode" in dockerignore_content, ".dockerignore should exclude .vscode/"

    def test_excludes_coverage_files(self, dockerignore_content: str) -> None:
        """Should exclude coverage files."""
        assert ".coverage" in dockerignore_content, ".dockerignore should exclude .coverage"


class TestDockerBuildPrerequisites:
    """Tests for files required for Docker build."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root path."""
        return Path(__file__).parent.parent.parent

    def test_pyproject_toml_exists(self, project_root: Path) -> None:
        """pyproject.toml should exist for dependency specification."""
        pyproject = project_root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml is required for Docker build"

    def test_uv_lock_exists(self, project_root: Path) -> None:
        """uv.lock should exist for reproducible builds."""
        uv_lock = project_root / "uv.lock"
        assert uv_lock.exists(), "uv.lock is required for Docker build"

    def test_backend_directory_exists(self, project_root: Path) -> None:
        """backend/ directory should exist."""
        backend = project_root / "backend"
        assert backend.exists(), "backend/ directory is required"
        assert backend.is_dir(), "backend should be a directory"

    def test_backend_api_server_exists(self, project_root: Path) -> None:
        """backend/api/server.py should exist (entrypoint)."""
        server = project_root / "backend" / "api" / "server.py"
        assert server.exists(), "backend/api/server.py is required as entrypoint"

    def test_config_directory_exists(self, project_root: Path) -> None:
        """config/ directory should exist for configuration files."""
        config = project_root / "config"
        assert config.exists(), "config/ directory should exist"


class TestFrontendDockerfile:
    """Tests for frontend Dockerfile structure and content."""

    @pytest.fixture
    def dockerfile_path(self) -> Path:
        """Get the path to the frontend Dockerfile."""
        return Path(__file__).parent.parent.parent / "docker" / "Dockerfile.frontend"

    @pytest.fixture
    def dockerfile_content(self, dockerfile_path: Path) -> str:
        """Read the Dockerfile content."""
        assert dockerfile_path.exists(), f"Dockerfile not found at {dockerfile_path}"
        return dockerfile_path.read_text()

    def test_dockerfile_exists(self, dockerfile_path: Path) -> None:
        """Dockerfile.frontend should exist in docker/ directory."""
        assert dockerfile_path.exists(), "docker/Dockerfile.frontend does not exist"

    def test_uses_node_20_alpine_base(self, dockerfile_content: str) -> None:
        """Should use node:20-alpine as base image."""
        assert "FROM node:20-alpine" in dockerfile_content, (
            "Dockerfile should use node:20-alpine base image"
        )

    def test_uses_multistage_build(self, dockerfile_content: str) -> None:
        """Should use multi-stage build for smaller image size."""
        from_count = dockerfile_content.count("FROM node:20-alpine")
        assert from_count >= 2, (
            "Dockerfile should use multi-stage build (at least 2 FROM instructions)"
        )

    def test_copies_package_files(self, dockerfile_content: str) -> None:
        """Should copy package.json and package-lock.json for dependency installation."""
        assert "package.json" in dockerfile_content, "Dockerfile should copy package.json"
        assert "package-lock.json" in dockerfile_content, "Dockerfile should copy package-lock.json"

    def test_runs_npm_ci(self, dockerfile_content: str) -> None:
        """Should run npm ci for clean dependency installation."""
        assert "npm ci" in dockerfile_content, "Dockerfile should run npm ci"

    def test_runs_npm_build(self, dockerfile_content: str) -> None:
        """Should run npm run build for Next.js build."""
        assert "npm run build" in dockerfile_content, "Dockerfile should run npm run build"

    def test_exposes_port_3000(self, dockerfile_content: str) -> None:
        """Should expose port 3000 for Next.js."""
        assert "EXPOSE 3000" in dockerfile_content, "Dockerfile should expose port 3000"

    def test_has_healthcheck(self, dockerfile_content: str) -> None:
        """Should have HEALTHCHECK instruction."""
        assert "HEALTHCHECK" in dockerfile_content, "Dockerfile should have HEALTHCHECK instruction"

    def test_cmd_runs_npm_start(self, dockerfile_content: str) -> None:
        """Should have CMD that runs npm start."""
        assert re.search(r"CMD.*npm.*start", dockerfile_content), "Dockerfile should run npm start"

    def test_sets_node_env_production(self, dockerfile_content: str) -> None:
        """Should set NODE_ENV to production."""
        assert "NODE_ENV=production" in dockerfile_content, (
            "Dockerfile should set NODE_ENV=production"
        )

    def test_sets_next_telemetry_disabled(self, dockerfile_content: str) -> None:
        """Should disable Next.js telemetry."""
        assert "NEXT_TELEMETRY_DISABLED" in dockerfile_content, (
            "Dockerfile should disable Next.js telemetry"
        )

    def test_creates_non_root_user(self, dockerfile_content: str) -> None:
        """Should create a non-root user for security."""
        assert "adduser" in dockerfile_content or "useradd" in dockerfile_content, (
            "Dockerfile should create a non-root user"
        )
        assert "USER" in dockerfile_content, "Dockerfile should switch to non-root user"

    def test_copies_next_build_output(self, dockerfile_content: str) -> None:
        """Should copy .next build output from builder stage."""
        assert ".next" in dockerfile_content, "Dockerfile should copy .next directory"

    def test_copies_public_directory(self, dockerfile_content: str) -> None:
        """Should copy public directory for static assets."""
        assert "/public" in dockerfile_content or "public" in dockerfile_content, (
            "Dockerfile should copy public directory"
        )


class TestFrontendBuildPrerequisites:
    """Tests for files required for frontend Docker build."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root path."""
        return Path(__file__).parent.parent.parent

    def test_frontend_directory_exists(self, project_root: Path) -> None:
        """frontend/ directory should exist."""
        frontend = project_root / "frontend"
        assert frontend.exists(), "frontend/ directory is required"
        assert frontend.is_dir(), "frontend should be a directory"

    def test_package_json_exists(self, project_root: Path) -> None:
        """frontend/package.json should exist."""
        package_json = project_root / "frontend" / "package.json"
        assert package_json.exists(), "frontend/package.json is required for Docker build"

    def test_package_lock_exists(self, project_root: Path) -> None:
        """frontend/package-lock.json should exist for reproducible builds."""
        package_lock = project_root / "frontend" / "package-lock.json"
        assert package_lock.exists(), "frontend/package-lock.json is required for Docker build"

    def test_next_config_exists(self, project_root: Path) -> None:
        """frontend/next.config.ts should exist."""
        next_config = project_root / "frontend" / "next.config.ts"
        assert next_config.exists(), "frontend/next.config.ts is required"

    def test_app_directory_exists(self, project_root: Path) -> None:
        """frontend/app/ directory should exist (Next.js App Router)."""
        app_dir = project_root / "frontend" / "app"
        assert app_dir.exists(), "frontend/app/ directory is required for Next.js"
        assert app_dir.is_dir(), "frontend/app should be a directory"


class TestDockerCompose:
    """Tests for docker-compose.yml structure and content using string matching.

    These tests validate the docker-compose.yml configuration without requiring
    the pyyaml dependency. They use regex and string matching to verify the
    expected configuration is present.
    """

    @pytest.fixture
    def compose_path(self) -> Path:
        """Get the path to docker-compose.yml."""
        return Path(__file__).parent.parent.parent / "docker-compose.yml"

    @pytest.fixture
    def compose_content(self, compose_path: Path) -> str:
        """Read the docker-compose.yml raw content."""
        assert compose_path.exists(), f"docker-compose.yml not found at {compose_path}"
        return compose_path.read_text()

    def test_compose_file_exists(self, compose_path: Path) -> None:
        """docker-compose.yml should exist in project root."""
        assert compose_path.exists(), "docker-compose.yml does not exist"

    def test_defines_services_section(self, compose_content: str) -> None:
        """Should have services section."""
        assert "services:" in compose_content, "docker-compose.yml should have services section"

    def test_defines_backend_service(self, compose_content: str) -> None:
        """Should define a backend service."""
        assert re.search(r"^\s+backend:", compose_content, re.MULTILINE), (
            "Should define backend service"
        )

    def test_defines_frontend_service(self, compose_content: str) -> None:
        """Should define a frontend service."""
        assert re.search(r"^\s+frontend:", compose_content, re.MULTILINE), (
            "Should define frontend service"
        )

    def test_backend_uses_dockerfile(self, compose_content: str) -> None:
        """Backend service should reference Dockerfile.backend."""
        assert "Dockerfile.backend" in compose_content, (
            "Backend should use docker/Dockerfile.backend"
        )

    def test_frontend_uses_dockerfile(self, compose_content: str) -> None:
        """Frontend service should reference Dockerfile.frontend."""
        assert "Dockerfile.frontend" in compose_content, (
            "Frontend should use docker/Dockerfile.frontend"
        )

    def test_backend_port_mapping(self, compose_content: str) -> None:
        """Backend should expose port 8000."""
        assert re.search(r'"?8000:8000"?', compose_content), (
            "Backend should expose port 8000"
        )

    def test_frontend_port_mapping(self, compose_content: str) -> None:
        """Frontend should expose port 3000."""
        assert re.search(r'"?3000:3000"?', compose_content), (
            "Frontend should expose port 3000"
        )

    def test_backend_has_volumes(self, compose_content: str) -> None:
        """Backend should have volume mounts defined."""
        # Look for volumes section under backend
        assert "volumes:" in compose_content, "Should define volumes"

    def test_backend_mounts_data_directory(self, compose_content: str) -> None:
        """Backend should mount data directory for persistence."""
        # Match patterns like ./data:/app/data or data:/app/data
        assert re.search(r"\.?/?data.*:/app/data", compose_content), (
            "Backend should mount data directory for SQLite/cache"
        )

    def test_backend_mounts_logs_directory(self, compose_content: str) -> None:
        """Backend should mount logs directory."""
        # Match patterns like ./logs:/app/logs or logs:/app/logs
        assert re.search(r"\.?/?logs.*:/app/logs", compose_content), (
            "Backend should mount logs directory"
        )

    def test_backend_has_env_file(self, compose_content: str) -> None:
        """Backend should use env_file for configuration."""
        assert "env_file:" in compose_content, "Backend should use env_file"

    def test_backend_has_healthcheck(self, compose_content: str) -> None:
        """Backend should have healthcheck configured."""
        assert "healthcheck:" in compose_content, "Should have healthcheck configured"
        # Backend healthcheck should reference port 8000 or /health
        assert "8000" in compose_content and "health" in compose_content.lower(), (
            "Backend healthcheck should test health endpoint"
        )

    def test_frontend_depends_on_backend(self, compose_content: str) -> None:
        """Frontend should depend on backend service."""
        assert "depends_on:" in compose_content, "Frontend should have depends_on"
        assert "backend:" in compose_content, "Frontend should depend on backend"

    def test_frontend_waits_for_healthy_backend(self, compose_content: str) -> None:
        """Frontend should wait for backend to be healthy."""
        assert "service_healthy" in compose_content, (
            "Frontend should wait for backend to be healthy"
        )

    def test_frontend_has_api_url_environment(self, compose_content: str) -> None:
        """Frontend should have NEXT_PUBLIC_API_URL pointing to backend."""
        assert "NEXT_PUBLIC_API_URL" in compose_content, (
            "Frontend should have NEXT_PUBLIC_API_URL environment variable"
        )
        # API URL should point to backend service
        assert re.search(r"NEXT_PUBLIC_API_URL.*backend", compose_content), (
            "API URL should point to backend service"
        )

    def test_frontend_has_ws_url_environment(self, compose_content: str) -> None:
        """Frontend should have NEXT_PUBLIC_WS_URL for WebSocket."""
        assert "NEXT_PUBLIC_WS_URL" in compose_content, (
            "Frontend should have NEXT_PUBLIC_WS_URL environment variable"
        )

    def test_defines_network(self, compose_content: str) -> None:
        """Should define a Docker network for inter-service communication."""
        assert "networks:" in compose_content, "docker-compose.yml should define networks"

    def test_services_use_network(self, compose_content: str) -> None:
        """Services should be connected to the defined network."""
        # Count networks: occurrences - should be at least 3 (definition + 2 services)
        network_count = compose_content.count("networks:")
        assert network_count >= 3, (
            f"Services should be connected to network (found {network_count}, expected at least 3)"
        )

    def test_backend_has_restart_policy(self, compose_content: str) -> None:
        """Backend should have restart policy configured."""
        assert "restart:" in compose_content, "Should have restart policy"
        assert re.search(r"restart:\s*(unless-stopped|always|on-failure)", compose_content), (
            "Should have proper restart policy"
        )

    def test_has_container_names(self, compose_content: str) -> None:
        """Services should have container names defined."""
        assert "container_name:" in compose_content, "Should define container names"
        assert "fluxhero-backend" in compose_content, "Backend should have named container"
        assert "fluxhero-frontend" in compose_content, "Frontend should have named container"

    def test_backend_overrides_cache_dir(self, compose_content: str) -> None:
        """Backend should override FLUXHERO_CACHE_DIR for container paths."""
        assert "FLUXHERO_CACHE_DIR" in compose_content, (
            "Backend should set FLUXHERO_CACHE_DIR"
        )

    def test_backend_overrides_log_file(self, compose_content: str) -> None:
        """Backend should override FLUXHERO_LOG_FILE for container paths."""
        assert "FLUXHERO_LOG_FILE" in compose_content, (
            "Backend should set FLUXHERO_LOG_FILE"
        )
