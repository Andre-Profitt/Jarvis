"""
Test suite for JARVIS Tools
===========================

Tests for all tools in the tools module.
"""

import asyncio
import pytest
import json
import tempfile
import os
from pathlib import Path

from tools import (
    tool_registry,
    WebSearchTool,
    FileManagerTool,
    CodeExecutorTool,
    DataProcessorTool,
    APIWrapperTool,
    ToolCategory,
    ToolStatus,
)


class TestToolRegistry:
    """Test the tool registry functionality"""

    def test_registry_initialization(self):
        """Test that tools are registered on import"""
        assert len(tool_registry.list_tools()) >= 5

        # Check specific tools are registered
        assert tool_registry.get("web_search") is not None
        assert tool_registry.get("file_manager") is not None
        assert tool_registry.get("code_executor") is not None
        assert tool_registry.get("data_processor") is not None
        assert tool_registry.get("api_wrapper") is not None

    def test_list_tools_by_category(self):
        """Test listing tools by category"""
        web_tools = tool_registry.list_tools(ToolCategory.WEB)
        assert "web_search" in web_tools
        assert "api_wrapper" in web_tools

        file_tools = tool_registry.list_tools(ToolCategory.FILE)
        assert "file_manager" in file_tools

    def test_search_tools(self):
        """Test searching for tools"""
        # Search by name
        results = tool_registry.search_tools("web")
        assert len(results) >= 1
        assert any(tool.metadata.name == "web_search" for tool in results)

        # Search by tag
        results = tool_registry.search_tools("search")
        assert len(results) >= 1

    def test_get_documentation(self):
        """Test getting tool documentation"""
        docs = tool_registry.get_documentation()
        assert isinstance(docs, dict)
        assert len(docs) > 0


class TestWebSearchTool:
    """Test the WebSearchTool"""

    @pytest.mark.asyncio
    async def test_web_search_basic(self):
        """Test basic web search functionality"""
        tool = WebSearchTool()

        result = await tool.execute(query="artificial intelligence", num_results=5)

        assert result.success
        assert "results" in result.data
        assert len(result.data["results"]) == 5
        assert result.data["query"] == "artificial intelligence"

    @pytest.mark.asyncio
    async def test_web_search_validation(self):
        """Test input validation"""
        tool = WebSearchTool()

        # Test missing query
        result = await tool.execute()
        assert not result.success
        assert "Query parameter is required" in result.error

        # Test empty query
        result = await tool.execute(query="")
        assert not result.success
        assert "Query cannot be empty" in result.error

        # Test invalid num_results
        result = await tool.execute(query="test", num_results=200)
        assert not result.success
        assert "num_results must be an integer between 1 and 100" in result.error

    @pytest.mark.asyncio
    async def test_web_search_caching(self):
        """Test search result caching"""
        tool = WebSearchTool()

        # First search
        result1 = await tool.execute(query="test query", use_cache=True)
        assert result1.success
        assert not result1.data.get("cached", False)

        # Second search (should be cached)
        result2 = await tool.execute(query="test query", use_cache=True)
        assert result2.success
        assert result2.data.get("cached", False)


class TestFileManagerTool:
    """Test the FileManagerTool"""

    @pytest.mark.asyncio
    async def test_file_operations(self):
        """Test basic file operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = FileManagerTool(base_path=temp_dir)

            # Test write
            result = await tool.execute(
                operation="write", path="test.txt", content="Hello, JARVIS!"
            )
            assert result.success
            assert result.data["operation"] == "write"

            # Test read
            result = await tool.execute(operation="read", path="test.txt")
            assert result.success
            assert result.data["content"] == "Hello, JARVIS!"

            # Test exists
            result = await tool.execute(operation="exists", path="test.txt")
            assert result.success
            assert result.data["exists"] is True

            # Test delete
            result = await tool.execute(operation="delete", path="test.txt")
            assert result.success
            assert result.data["deleted"] is True

    @pytest.mark.asyncio
    async def test_directory_operations(self):
        """Test directory operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = FileManagerTool(base_path=temp_dir)

            # Create directory
            result = await tool.execute(operation="create_dir", path="test_dir")
            assert result.success

            # List directory
            result = await tool.execute(operation="list", path=".")
            assert result.success
            assert any(f["name"] == "test_dir" for f in result.data["files"])

    @pytest.mark.asyncio
    async def test_json_handling(self):
        """Test JSON file handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = FileManagerTool(base_path=temp_dir)

            test_data = {"name": "JARVIS", "version": "1.0"}

            # Write JSON
            result = await tool.execute(
                operation="write", path="data.json", content=test_data
            )
            assert result.success

            # Read JSON
            result = await tool.execute(operation="read", path="data.json")
            assert result.success
            assert result.data["content"] == test_data


class TestCodeExecutorTool:
    """Test the CodeExecutorTool"""

    @pytest.mark.asyncio
    async def test_python_execution(self):
        """Test Python code execution"""
        tool = CodeExecutorTool()

        result = await tool.execute(
            code="print('Hello from JARVIS!')\nresult = 2 + 2\nprint(f'Result: {result}')",
            language="python",
        )

        assert result.success
        assert "output" in result.data
        assert "Hello from JARVIS!" in result.data["output"][0]
        assert "Result: 4" in result.data["output"][1]

    @pytest.mark.asyncio
    async def test_code_validation(self):
        """Test code validation"""
        tool = CodeExecutorTool()

        # Test syntax error
        result = await tool.execute(
            code="print('missing parenthesis'", language="python"
        )
        assert not result.success
        assert "syntax error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_input_data(self):
        """Test code execution with input data"""
        tool = CodeExecutorTool()

        result = await tool.execute(
            code="total = sum(input_data['numbers'])\nprint(f'Total: {total}')",
            language="python",
            input_data={"numbers": [1, 2, 3, 4, 5]},
        )

        assert result.success
        assert "Total: 15" in result.data["output"][0]

    @pytest.mark.asyncio
    async def test_dangerous_code_detection(self):
        """Test dangerous code detection"""
        tool = CodeExecutorTool()

        result = await tool.execute(
            code="import os\nos.system('rm -rf /')", language="python"
        )
        assert not result.success
        assert "dangerous" in result.error.lower()


class TestDataProcessorTool:
    """Test the DataProcessorTool"""

    @pytest.mark.asyncio
    async def test_data_transform(self):
        """Test data transformation"""
        tool = DataProcessorTool()

        test_data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]

        result = await tool.execute(
            operation="transform",
            data=test_data,
            transformations=[
                {
                    "type": "add_field",
                    "field": "adult",
                    "value": "lambda x: x['age'] >= 18",
                }
            ],
        )

        assert result.success
        assert all(item["adult"] for item in result.data["data"])

    @pytest.mark.asyncio
    async def test_data_filter(self):
        """Test data filtering"""
        tool = DataProcessorTool()

        test_data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]

        result = await tool.execute(
            operation="filter",
            data=test_data,
            conditions=[{"field": "age", "operator": "gt", "value": 25}],
        )

        assert result.success
        assert result.data["filtered_count"] == 2
        assert all(item["age"] > 25 for item in result.data["data"])

    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test statistical calculations"""
        tool = DataProcessorTool()

        result = await tool.execute(
            operation="statistics",
            data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            metrics=["mean", "median", "std_dev"],
        )

        assert result.success
        assert result.data["metrics"]["mean"] == 5.5
        assert result.data["metrics"]["median"] == 5.5
        assert "std_dev" in result.data["metrics"]

    @pytest.mark.asyncio
    async def test_data_conversion(self):
        """Test format conversion"""
        tool = DataProcessorTool()

        test_data = [{"id": 1, "name": "Test"}]

        # Convert to CSV
        result = await tool.execute(
            operation="convert", data=test_data, from_format="python", to_format="csv"
        )

        assert result.success
        assert "id,name" in result.data["data"]
        assert "1,Test" in result.data["data"]


class TestAPIWrapperTool:
    """Test the APIWrapperTool"""

    @pytest.mark.asyncio
    async def test_get_request(self):
        """Test GET request"""
        tool = APIWrapperTool()

        # Using a public test API
        result = await tool.execute(
            method="GET", url="https://jsonplaceholder.typicode.com/posts/1"
        )

        assert result.success
        assert result.data["status_code"] == 200
        assert "json" in result.data
        assert result.data["json"]["id"] == 1

        await tool.close()

    @pytest.mark.asyncio
    async def test_post_request(self):
        """Test POST request"""
        tool = APIWrapperTool()

        result = await tool.execute(
            method="POST",
            url="https://jsonplaceholder.typicode.com/posts",
            json={"title": "Test Post", "body": "Test body", "userId": 1},
        )

        assert result.success
        assert result.data["status_code"] == 201
        assert result.data["json"]["title"] == "Test Post"

        await tool.close()

    @pytest.mark.asyncio
    async def test_url_validation(self):
        """Test URL validation"""
        tool = APIWrapperTool()

        # Test missing URL
        result = await tool.execute(method="GET")
        assert not result.success
        assert "URL parameter is required" in result.error

        # Test invalid URL
        result = await tool.execute(method="GET", url="not-a-url")
        assert not result.success
        assert "URL must start with http://" in result.error

        await tool.close()

    @pytest.mark.asyncio
    async def test_authentication(self):
        """Test authentication headers"""
        tool = APIWrapperTool()

        result = await tool.execute(
            method="GET",
            url="https://httpbin.org/headers",
            auth={"type": "bearer", "token": "test-token"},
        )

        assert result.success
        headers = result.data["json"]["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token"

        await tool.close()


# Integration Tests


class TestToolIntegration:
    """Test integration between multiple tools"""

    @pytest.mark.asyncio
    async def test_search_and_process(self):
        """Test searching and then processing results"""
        search_tool = WebSearchTool()
        processor_tool = DataProcessorTool()

        # Search for data
        search_result = await search_tool.execute(
            query="python programming", num_results=5
        )
        assert search_result.success

        # Process search results
        process_result = await processor_tool.execute(
            operation="transform",
            data=search_result.data["results"],
            transformations=[
                {
                    "type": "add_field",
                    "field": "domain",
                    "value": "lambda x: x['url'].split('/')[2] if 'url' in x else 'unknown'",
                }
            ],
        )
        assert process_result.success
        assert all("domain" in item for item in process_result.data["data"])

    @pytest.mark.asyncio
    async def test_file_and_code_execution(self):
        """Test writing code to file and executing it"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_tool = FileManagerTool(base_path=temp_dir)
            code_tool = CodeExecutorTool()

            # Write Python script
            code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = [fibonacci(i) for i in range(10)]
print(f"Fibonacci sequence: {result}")
"""

            write_result = await file_tool.execute(
                operation="write", path="fibonacci.py", content=code
            )
            assert write_result.success

            # Read and execute
            read_result = await file_tool.execute(operation="read", path="fibonacci.py")
            assert read_result.success

            exec_result = await code_tool.execute(
                code=read_result.data["content"], language="python"
            )
            assert exec_result.success
            assert "Fibonacci sequence:" in exec_result.data["output"][0]


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
