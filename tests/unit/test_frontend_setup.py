"""
Unit tests for Frontend Setup (Task 3: React + Next.js + TypeScript)

Validates:
- Next.js project structure exists
- TypeScript configuration is correct
- API integration utility is implemented
- Required dependencies are installed
- Configuration files are present
"""

import json
from pathlib import Path


class TestFrontendSetup:
    """Test suite for frontend project setup"""

    @staticmethod
    def get_frontend_path():
        """Get the path to the frontend directory"""
        return Path(__file__).parent.parent.parent / "fluxhero" / "frontend"

    def test_frontend_directory_exists(self):
        """Verify frontend directory exists"""
        frontend_path = self.get_frontend_path()
        assert frontend_path.exists(), "Frontend directory should exist"
        assert frontend_path.is_dir(), "Frontend should be a directory"

    def test_package_json_exists(self):
        """Verify package.json exists and is valid"""
        package_json_path = self.get_frontend_path() / "package.json"
        assert package_json_path.exists(), "package.json should exist"

        with open(package_json_path) as f:
            package_data = json.load(f)

        assert package_data["name"] == "fluxhero-frontend", "Package name should be fluxhero-frontend"
        assert "description" in package_data, "Package should have description"
        assert package_data["description"] == "FluxHero Trading System Frontend"

    def test_next_js_dependencies(self):
        """Verify Next.js and React dependencies are installed"""
        package_json_path = self.get_frontend_path() / "package.json"

        with open(package_json_path) as f:
            package_data = json.load(f)

        dependencies = package_data.get("dependencies", {})

        assert "next" in dependencies, "Next.js should be in dependencies"
        assert "react" in dependencies, "React should be in dependencies"
        assert "react-dom" in dependencies, "React DOM should be in dependencies"

    def test_typescript_dependencies(self):
        """Verify TypeScript dependencies are installed"""
        package_json_path = self.get_frontend_path() / "package.json"

        with open(package_json_path) as f:
            package_data = json.load(f)

        dev_dependencies = package_data.get("devDependencies", {})

        assert "typescript" in dev_dependencies, "TypeScript should be in devDependencies"
        assert "@types/react" in dev_dependencies, "React types should be in devDependencies"
        assert "@types/node" in dev_dependencies, "Node types should be in devDependencies"

    def test_npm_scripts(self):
        """Verify required npm scripts are configured"""
        package_json_path = self.get_frontend_path() / "package.json"

        with open(package_json_path) as f:
            package_data = json.load(f)

        scripts = package_data.get("scripts", {})

        assert "dev" in scripts, "dev script should exist"
        assert "build" in scripts, "build script should exist"
        assert "start" in scripts, "start script should exist"
        assert "lint" in scripts, "lint script should exist"

    def test_tsconfig_exists(self):
        """Verify TypeScript configuration exists"""
        tsconfig_path = self.get_frontend_path() / "tsconfig.json"
        assert tsconfig_path.exists(), "tsconfig.json should exist"

        with open(tsconfig_path) as f:
            tsconfig = json.load(f)

        compiler_options = tsconfig.get("compilerOptions", {})
        assert compiler_options.get("strict") is True, "Strict mode should be enabled"
        assert compiler_options.get("jsx") == "preserve", "JSX should be set to preserve"
        assert "@/*" in compiler_options.get("paths", {}), "Path aliases should be configured"

    def test_next_config_exists(self):
        """Verify Next.js configuration exists"""
        next_config_path = self.get_frontend_path() / "next.config.ts"
        assert next_config_path.exists(), "next.config.ts should exist"

        with open(next_config_path) as f:
            content = f.read()

        assert "NextConfig" in content, "Should import NextConfig type"
        assert "rewrites" in content, "Should configure API rewrites"
        assert "/api/:path*" in content, "Should proxy API routes"
        assert "/ws/:path*" in content, "Should proxy WebSocket routes"

    def test_api_utility_exists(self):
        """Verify API utility file exists"""
        api_path = self.get_frontend_path() / "utils" / "api.ts"
        assert api_path.exists(), "API utility should exist"

        with open(api_path) as f:
            content = f.read()

        # Check for API client class
        assert "class ApiClient" in content, "Should have ApiClient class"
        assert "apiClient" in content, "Should export apiClient instance"

        # Check for API methods
        assert "getPositions" in content, "Should have getPositions method"
        assert "getTrades" in content, "Should have getTrades method"
        assert "getAccountInfo" in content, "Should have getAccountInfo method"
        assert "getSystemStatus" in content, "Should have getSystemStatus method"
        assert "runBacktest" in content, "Should have runBacktest method"
        assert "connectPriceWebSocket" in content, "Should have WebSocket connection method"

    def test_api_types_defined(self):
        """Verify TypeScript types are defined for API responses"""
        api_path = self.get_frontend_path() / "utils" / "api.ts"

        with open(api_path) as f:
            content = f.read()

        # Check for TypeScript interfaces
        assert "interface Position" in content, "Should define Position interface"
        assert "interface Trade" in content, "Should define Trade interface"
        assert "interface AccountInfo" in content, "Should define AccountInfo interface"
        assert "interface SystemStatus" in content, "Should define SystemStatus interface"
        assert "interface BacktestConfig" in content, "Should define BacktestConfig interface"
        assert "interface BacktestResult" in content, "Should define BacktestResult interface"

    def test_app_directory_structure(self):
        """Verify Next.js app directory structure"""
        frontend_path = self.get_frontend_path()

        app_dir = frontend_path / "app"
        assert app_dir.exists(), "app directory should exist"

        layout_file = app_dir / "layout.tsx"
        assert layout_file.exists(), "Root layout should exist"

        page_file = app_dir / "page.tsx"
        assert page_file.exists(), "Home page should exist"

        globals_css = app_dir / "globals.css"
        assert globals_css.exists(), "Global CSS should exist"

    def test_placeholder_pages_directory(self):
        """Verify pages directory exists for future implementation"""
        pages_dir = self.get_frontend_path() / "pages"
        assert pages_dir.exists(), "pages directory should exist"

    def test_components_directory(self):
        """Verify components directory exists"""
        components_dir = self.get_frontend_path() / "components"
        assert components_dir.exists(), "components directory should exist"

    def test_env_example_exists(self):
        """Verify environment variable example file exists"""
        env_example = self.get_frontend_path() / ".env.local.example"
        assert env_example.exists(), ".env.local.example should exist"

        with open(env_example) as f:
            content = f.read()

        assert "NEXT_PUBLIC_API_URL" in content, "Should have API URL configuration"
        assert "NEXT_PUBLIC_WS_URL" in content, "Should have WebSocket URL configuration"

    def test_readme_exists(self):
        """Verify README documentation exists"""
        readme_path = self.get_frontend_path() / "README.md"
        assert readme_path.exists(), "Frontend README should exist"

        with open(readme_path) as f:
            content = f.read()

        assert "FluxHero Frontend" in content, "Should have project title"
        assert "Next.js" in content, "Should mention Next.js"
        assert "TypeScript" in content, "Should mention TypeScript"
        assert "API Integration" in content, "Should mention API integration"
