#!/usr/bin/env python3
"""
JARVIS MCP Server for Claude Desktop
Provides system-wide capabilities to Claude
"""

import asyncio
import json
import sys
import logging
from typing import Dict, Any, List, Optional
import subprocess
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class JARVISMCPServer:
    """MCP Server implementation for JARVIS capabilities"""

    def __init__(self):
        logger.info("Initializing JARVIS MCP Server")
        self.capabilities = {
            "name": "jarvis",
            "version": "1.0.0",
            "vendor": "JARVIS-ECOSYSTEM",
            "capabilities": {
                "resources": {"listResources": True, "readResource": True},
                "tools": {"listTools": True, "callTool": True},
                "prompts": {"listPrompts": True, "getPrompt": True},
            },
        }

        self.tools = {
            "execute_command": {
                "name": "execute_command",
                "description": "Execute system commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"}
                    },
                    "required": ["command"]
                },
            },
            "read_file": {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                },
            },
            "write_file": {
                "name": "write_file",
                "description": "Write file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"]
                },
            },
            "search_codebase": {
                "name": "search_codebase",
                "description": "Search through codebase",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                },
            },
            "jarvis_think": {
                "name": "jarvis_think",
                "description": "Let JARVIS think and reason about complex problems",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "problem": {"type": "string", "description": "Problem to solve"}
                    },
                    "required": ["problem"]
                },
            },
        }

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        method = request.get("method", "")
        params = request.get("params", {})
        logger.debug(f"Handling request: method={method}, params={params}")

        if method == "initialize":
            logger.info("Processing initialize request")
            return {"result": self.capabilities}

        elif method == "tools/list":
            logger.info("Listing tools")
            return {"result": {"tools": list(self.tools.values())}}

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            logger.info(f"Calling tool: {tool_name} with args: {tool_args}")

            if tool_name == "execute_command":
                result = subprocess.run(
                    tool_args["command"], shell=True, capture_output=True, text=True
                )
                return {
                    "result": {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                    }
                }

            elif tool_name == "read_file":
                try:
                    with open(tool_args["path"], "r") as f:
                        return {"result": {"content": f.read()}}
                except Exception as e:
                    return {"error": {"code": -32603, "message": str(e)}}

            elif tool_name == "write_file":
                try:
                    with open(tool_args["path"], "w") as f:
                        f.write(tool_args["content"])
                    return {"result": {"success": True}}
                except Exception as e:
                    return {"error": {"code": -32603, "message": str(e)}}

            elif tool_name == "search_codebase":
                # Check if ripgrep is available
                try:
                    subprocess.run(["rg", "--version"], capture_output=True, check=True)
                    # Use ripgrep for fast searching
                    result = subprocess.run(
                        ["rg", "-i", tool_args["query"], "."],
                        capture_output=True,
                        text=True,
                        cwd=os.environ.get("JARVIS_HOME", ".")
                    )
                    return {"result": {"matches": result.stdout}}
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback to grep if ripgrep not available
                    logger.warning("ripgrep not found, falling back to grep")
                    result = subprocess.run(
                        ["grep", "-r", "-i", tool_args["query"], "."],
                        capture_output=True,
                        text=True,
                        cwd=os.environ.get("JARVIS_HOME", ".")
                    )
                    return {"result": {"matches": result.stdout}}

            elif tool_name == "jarvis_think":
                # This is where JARVIS's reasoning happens
                return {
                    "result": {
                        "thoughts": f"Analyzing: {tool_args['problem']}",
                        "approach": "Breaking down the problem into steps...",
                        "solution": "Here's my reasoned approach...",
                    }
                }

            else:
                return {"error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}

        return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting JARVIS MCP Server")
        
        try:
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            transport, _ = await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
            logger.info("Connected to stdin")
        except Exception as e:
            logger.error(f"Failed to connect to stdin: {e}")
            return

        while True:
            try:
                # Read JSON-RPC request
                logger.debug("Waiting for request...")
                line = await reader.readline()
                if not line:
                    logger.info("No more input, shutting down")
                    break

                line_str = line.decode().strip()
                logger.debug(f"Received: {line_str}")
                
                request = json.loads(line_str)
                response = await self.handle_request(request)

                # Send JSON-RPC response
                response["jsonrpc"] = "2.0"
                response["id"] = request.get("id")

                response_str = json.dumps(response)
                logger.debug(f"Sending response: {response_str}")
                
                sys.stdout.write(response_str + "\n")
                sys.stdout.flush()

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error"},
                    "id": None,
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)},
                    "id": request.get("id") if "request" in locals() else None,
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    logger.info("JARVIS MCP Server starting up")
    server = JARVISMCPServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}", exc_info=True)
