#!/usr/bin/env python3
"""
Autonomous Tool Creation System for JARVIS
Creates, tests, and deploys new MCP tools automatically
"""

import ast
import inspect
import textwrap
import black
import pytest
import docker
import yaml
import json
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import git
from github import Github
import openai
from anthropic import Anthropic
import requests
from jinja2 import Template

@dataclass
class ToolSpecification:
    """Specification for a new tool"""
    name: str
    description: str
    category: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    dependencies: List[str]
    api_endpoints: Optional[List[str]]
    authentication: Optional[Dict[str, Any]]
    rate_limits: Optional[Dict[str, Any]]

class AutonomousToolCreator:
    """Creates new tools automatically based on needs"""
    
    def __init__(self, storage_path: str = "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/tools"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.tool_registry = {}
        self.code_generator = AdvancedCodeGenerator()
        self.test_generator = TestGenerator()
        self.mcp_integrator = MCPIntegrator()
        self.deployment_system = ToolDeploymentSystem()
        
        # Tool creation strategies
        self.creation_strategies = {
            "api_wrapper": self._create_api_wrapper,
            "web_scraper": self._create_web_scraper,
            "data_processor": self._create_data_processor,
            "system_integration": self._create_system_integration,
            "ml_pipeline": self._create_ml_pipeline,
            "automation_script": self._create_automation_script
        }
    
    async def create_tool_from_need(self, need_description: str) -> Dict[str, Any]:
        """Create a tool based on a described need"""
        
        # Analyze need and generate specification
        spec = await self._analyze_and_specify(need_description)
        
        # Determine creation strategy
        strategy = await self._select_strategy(spec)
        
        # Create tool using selected strategy
        tool = await self.creation_strategies[strategy](spec)
        
        # Test tool
        test_results = await self._test_tool(tool)
        
        if test_results["passed"]:
            # Deploy tool
            deployment = await self.deployment_system.deploy(tool)
            
            # Register tool
            self.tool_registry[spec.name] = {
                "specification": spec,
                "implementation": tool,
                "tests": test_results,
                "deployment": deployment
            }
            
            return {
                "success": True,
                "tool": tool,
                "deployment": deployment
            }
        else:
            # Iterate on design
            improved_tool = await self._improve_tool(tool, test_results)
            return await self.create_tool_from_need(need_description)
    
    async def _analyze_and_specify(self, need_description: str) -> ToolSpecification:
        """Analyze need and create tool specification"""
        
        # Use AI to understand the need
        analysis_prompt = f"""
        Analyze this tool need and create a specification:
        
        Need: {need_description}
        
        Provide:
        1. Tool name (snake_case)
        2. Clear description
        3. Category (api_wrapper, web_scraper, data_processor, etc.)
        4. Required inputs with types
        5. Expected outputs with types
        6. Dependencies needed
        7. Any API endpoints involved
        8. Authentication requirements
        """
        
        # Get specification from AI
        spec_data = await self._get_ai_response(analysis_prompt)
        
        return ToolSpecification(**spec_data)
    
    async def _create_api_wrapper(self, spec: ToolSpecification) -> Dict[str, Any]:
        """Create an API wrapper tool"""
        
        # Discover API details
        api_details = await self._discover_api_details(spec.api_endpoints[0])
        
        # Generate wrapper code
        code = await self.code_generator.generate_api_wrapper(spec, api_details)
        
        # Create MCP integration
        mcp_tool = await self.mcp_integrator.create_mcp_tool(spec, code)
        
        # Generate tests
        tests = await self.test_generator.generate_api_tests(spec, code)
        
        return {
            "type": "api_wrapper",
            "specification": spec,
            "code": code,
            "mcp_integration": mcp_tool,
            "tests": tests,
            "api_details": api_details
        }
    
    async def _discover_api_details(self, api_endpoint: str) -> Dict[str, Any]:
        """Automatically discover API details"""
        
        details = {
            "base_url": api_endpoint,
            "endpoints": [],
            "authentication": None,
            "rate_limits": None
        }
        
        # Try to find OpenAPI/Swagger spec
        openapi_urls = [
            f"{api_endpoint}/openapi.json",
            f"{api_endpoint}/swagger.json",
            f"{api_endpoint}/api-docs",
            f"{api_endpoint}/v1/openapi.json"
        ]
        
        for url in openapi_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            openapi_spec = await response.json()
                            details["openapi"] = openapi_spec
                            details["endpoints"] = self._parse_openapi_endpoints(openapi_spec)
                            break
            except:
                continue
        
        # If no OpenAPI, try to discover through other means
        if not details.get("openapi"):
            details["endpoints"] = await self._discover_endpoints_heuristically(api_endpoint)
        
        return details

class AdvancedCodeGenerator:
    """Generates sophisticated code for tools"""
    
    def __init__(self):
        self.templates = self._load_code_templates()
        self.ai_model = "claude-opus-4"
        
    async def generate_api_wrapper(self, spec: ToolSpecification, 
                                  api_details: Dict[str, Any]) -> str:
        """Generate API wrapper code"""
        
        template = self.templates["api_wrapper"]
        
        # Prepare template variables
        context = {
            "tool_name": spec.name,
            "description": spec.description,
            "base_url": api_details["base_url"],
            "endpoints": api_details["endpoints"],
            "auth_type": api_details.get("authentication", {}).get("type", "none"),
            "rate_limits": spec.rate_limits
        }
        
        # Generate base code from template
        base_code = template.render(**context)
        
        # Enhance with AI
        enhanced_code = await self._enhance_code_with_ai(base_code, spec)
        
        # Format code
        formatted_code = black.format_str(enhanced_code, mode=black.Mode())
        
        return formatted_code
    
    async def generate_complex_tool(self, spec: ToolSpecification) -> str:
        """Generate code for complex multi-component tools"""
        
        components = await self._identify_components(spec)
        
        code_parts = []
        for component in components:
            component_code = await self._generate_component(component, spec)
            code_parts.append(component_code)
        
        # Combine components
        full_code = self._combine_components(code_parts, spec)
        
        # Add error handling, logging, etc.
        enhanced_code = await self._add_production_features(full_code)
        
        return enhanced_code
    
    def _load_code_templates(self) -> Dict[str, Template]:
        """Load code generation templates"""
        
        templates = {}
        
        # API Wrapper Template
        templates["api_wrapper"] = Template("""
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import logging
from functools import wraps
import time

class {{ tool_name }}:
    '''{{ description }}'''
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "{{ base_url }}"
        self.session = None
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        {% if rate_limits %}
        self.rate_limiter = RateLimiter(
            calls={{ rate_limits.calls }},
            period={{ rate_limits.period }}
        )
        {% endif %}
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        {% if auth_type == "bearer" %}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        {% elif auth_type == "api_key" %}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        {% endif %}
        return headers
    
    {% for endpoint in endpoints %}
    async def {{ endpoint.name }}(self, {% for param in endpoint.params %}{{ param.name }}: {{ param.type }}{% if not loop.last %}, {% endif %}{% endfor %}) -> {{ endpoint.return_type }}:
        '''{{ endpoint.description }}'''
        {% if rate_limits %}
        await self.rate_limiter.acquire()
        {% endif %}
        
        url = f"{self.base_url}{{ endpoint.path }}"
        {% if endpoint.method == "GET" %}
        params = {
            {% for param in endpoint.params if param.in == "query" %}
            "{{ param.name }}": {{ param.name }},
            {% endfor %}
        }
        async with self.session.get(url, headers=self._get_headers(), params=params) as response:
        {% else %}
        data = {
            {% for param in endpoint.params if param.in == "body" %}
            "{{ param.name }}": {{ param.name }},
            {% endfor %}
        }
        async with self.session.{{ endpoint.method.lower() }}(url, headers=self._get_headers(), json=data) as response:
        {% endif %}
            response.raise_for_status()
            return await response.json()
    {% endfor %}
""")
        
        return templates

class MCPIntegrator:
    """Integrates tools with MCP (Model Context Protocol)"""
    
    def __init__(self):
        self.mcp_server_path = "/Users/andreprofitt/CloudAI/mcp-servers"
        
    async def create_mcp_tool(self, spec: ToolSpecification, 
                             implementation: str) -> Dict[str, Any]:
        """Create MCP-compatible tool"""
        
        # Generate MCP server code
        mcp_server = await self._generate_mcp_server(spec, implementation)
        
        # Create package structure
        package_path = Path(self.mcp_server_path) / spec.name
        package_path.mkdir(exist_ok=True)
        
        # Write files
        (package_path / "server.py").write_text(mcp_server)
        (package_path / "__init__.py").write_text("")
        
        # Create package.json for MCP
        package_json = {
            "name": f"@jarvis/{spec.name}",
            "version": "1.0.0",
            "description": spec.description,
            "mcp": {
                "server": {
                    "command": "python",
                    "args": ["-m", spec.name, "server"]
                }
            }
        }
        
        (package_path / "package.json").write_text(json.dumps(package_json, indent=2))
        
        # Generate tool manifest
        manifest = await self._generate_tool_manifest(spec)
        
        return {
            "package_path": str(package_path),
            "manifest": manifest,
            "mcp_server": mcp_server
        }
    
    async def _generate_mcp_server(self, spec: ToolSpecification, 
                                  implementation: str) -> str:
        """Generate MCP server wrapper"""
        
        server_code = f"""
import asyncio
import json
from typing import Dict, Any
import sys

# Import the tool implementation
{implementation}

class MCPServer:
    def __init__(self):
        self.tool = {spec.name}()
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        method = request.get("method")
        params = request.get("params", {})
        
        if hasattr(self.tool, method):
            result = await getattr(self.tool, method)(**params)
            return {{"result": result}}
        else:
            return {{"error": f"Method {{method}} not found"}}
    
    async def run(self):
        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
                
            try:
                request = json.loads(line)
                response = await self.handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()
            except Exception as e:
                error_response = {{"error": str(e)}}
                print(json.dumps(error_response))
                sys.stdout.flush()

if __name__ == "__main__":
    server = MCPServer()
    asyncio.run(server.run())
"""
        
        return server_code

class TestGenerator:
    """Generates comprehensive tests for tools"""
    
    async def generate_api_tests(self, spec: ToolSpecification, 
                                code: str) -> str:
        """Generate tests for API wrapper"""
        
        test_code = f"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import aiohttp
from {spec.name} import {spec.name}

@pytest.fixture
async def tool():
    async with {spec.name}(api_key="test_key") as tool:
        yield tool

@pytest.fixture
def mock_response():
    mock = AsyncMock()
    mock.status = 200
    mock.json = AsyncMock(return_value={{"result": "success"}})
    mock.raise_for_status = AsyncMock()
    return mock

class Test{spec.name}:
    @pytest.mark.asyncio
    async def test_initialization(self):
        tool = {spec.name}(api_key="test_key")
        assert tool.api_key == "test_key"
        assert tool.base_url == "{spec.api_endpoints[0] if spec.api_endpoints else ''}"
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with {spec.name}() as tool:
            assert tool.session is not None
            assert isinstance(tool.session, aiohttp.ClientSession)
    
    # Add specific endpoint tests
    {await self._generate_endpoint_tests(spec)}
    
    @pytest.mark.asyncio
    async def test_error_handling(self, tool, mock_response):
        mock_response.status = 500
        mock_response.raise_for_status.side_effect = aiohttp.ClientError()
        
        with patch.object(tool.session, 'get', return_value=mock_response):
            with pytest.raises(aiohttp.ClientError):
                await tool.some_endpoint()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, tool):
        # Test rate limiting if applicable
        pass
"""
        
        return test_code

class ToolDeploymentSystem:
    """Deploys tools to various environments"""
    
    def __init__(self):
        self.deployment_strategies = {
            "local": self._deploy_local,
            "docker": self._deploy_docker,
            "cloud_function": self._deploy_cloud_function,
            "kubernetes": self._deploy_kubernetes
        }
    
    async def deploy(self, tool: Dict[str, Any], 
                    target: str = "local") -> Dict[str, Any]:
        """Deploy tool to specified target"""
        
        strategy = self.deployment_strategies.get(target, self._deploy_local)
        return await strategy(tool)
    
    async def _deploy_local(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy tool locally"""
        
        # Install in local MCP servers
        tool_path = Path(tool["mcp_integration"]["package_path"])
        
        # Update Claude Desktop config
        config_path = Path.home() / ".config" / "claude" / "claude_desktop_config.json"
        
        if config_path.exists():
            config = json.loads(config_path.read_text())
        else:
            config = {"mcpServers": {}}
        
        config["mcpServers"][tool["specification"].name] = {
            "command": "python",
            "args": ["-m", tool["specification"].name, "server"],
            "cwd": str(tool_path)
        }
        
        config_path.write_text(json.dumps(config, indent=2))
        
        return {
            "status": "deployed",
            "location": "local",
            "path": str(tool_path)
        }
    
    async def _deploy_docker(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy tool as Docker container"""
        
        # Generate Dockerfile
        dockerfile_content = f"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "{tool['specification'].name}", "server"]
"""
        
        tool_path = Path(tool["mcp_integration"]["package_path"])
        (tool_path / "Dockerfile").write_text(dockerfile_content)
        
        # Build Docker image
        client = docker.from_env()
        image_name = f"jarvis-tool-{tool['specification'].name}"
        
        image, logs = client.images.build(
            path=str(tool_path),
            tag=f"{image_name}:latest"
        )
        
        # Run container
        container = client.containers.run(
            image_name,
            detach=True,
            auto_remove=True,
            ports={'8080/tcp': None}
        )
        
        return {
            "status": "deployed",
            "location": "docker",
            "container_id": container.id,
            "image": image_name
        }

class DynamicToolDiscovery:
    """Discovers and integrates existing tools dynamically"""
    
    def __init__(self):
        self.discovered_tools = {}
        self.integration_strategies = {
            "github": self._integrate_github_tool,
            "pypi": self._integrate_pypi_tool,
            "npm": self._integrate_npm_tool,
            "api": self._integrate_api_tool
        }
    
    async def discover_tools_for_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Discover existing tools for a capability"""
        
        discovered = []
        
        # Search GitHub
        github_tools = await self._search_github(capability)
        discovered.extend(github_tools)
        
        # Search PyPI
        pypi_tools = await self._search_pypi(capability)
        discovered.extend(pypi_tools)
        
        # Search npm
        npm_tools = await self._search_npm(capability)
        discovered.extend(npm_tools)
        
        # Rank by relevance and quality
        ranked_tools = await self._rank_tools(discovered, capability)
        
        return ranked_tools[:10]  # Top 10 tools
    
    async def integrate_external_tool(self, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate an external tool into JARVIS"""
        
        source = tool_info["source"]
        strategy = self.integration_strategies.get(source)
        
        if strategy:
            return await strategy(tool_info)
        else:
            raise ValueError(f"Unknown tool source: {source}")
    
    async def _search_github(self, capability: str) -> List[Dict[str, Any]]:
        """Search GitHub for relevant tools"""
        
        g = Github()
        
        # Search repositories
        query = f"{capability} language:python stars:>10"
        repos = g.search_repositories(query=query, sort="stars", order="desc")
        
        tools = []
        for repo in repos[:20]:
            tool_info = {
                "name": repo.name,
                "description": repo.description,
                "url": repo.html_url,
                "stars": repo.stargazers_count,
                "source": "github",
                "language": repo.language,
                "topics": repo.get_topics()
            }
            
            # Check if it has tool-like structure
            if await self._is_tool_repository(repo):
                tools.append(tool_info)
        
        return tools

# Example usage
async def demonstrate_tool_creation():
    """Demonstrate autonomous tool creation"""
    
    creator = AutonomousToolCreator()
    
    # Example: Need a tool to analyze stock market data
    need = """
    I need a tool that can:
    1. Fetch real-time stock prices
    2. Calculate technical indicators (RSI, MACD, etc.)
    3. Generate trading signals
    4. Create visualizations
    5. Send alerts when conditions are met
    """
    
    print("🔧 Creating tool based on need...")
    result = await creator.create_tool_from_need(need)
    
    if result["success"]:
        print(f"✅ Tool created successfully!")
        print(f"   Name: {result['tool']['specification'].name}")
        print(f"   Type: {result['tool']['type']}")
        print(f"   Deployed to: {result['deployment']['location']}")
    
    # Example: Discover existing tools
    discovery = DynamicToolDiscovery()
    print("\n🔍 Discovering tools for 'data visualization'...")
    tools = await discovery.discover_tools_for_capability("data visualization")
    
    print(f"Found {len(tools)} relevant tools:")
    for tool in tools[:5]:
        print(f"   - {tool['name']}: {tool['description']}")

if __name__ == "__main__":
    asyncio.run(demonstrate_tool_creation())