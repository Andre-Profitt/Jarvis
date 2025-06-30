# MCP Integration Guide for Claude Code

## Overview

Claude Code **DOES** have access to MCP (Model Context Protocol) servers! This means it can potentially access your:
- mem0/LangChain long-term memory system
- 30TB Google Storage
- Any other MCP servers you have configured

## Current Status

### What Claude Code Can Access:
1. **MCP Servers Already Configured** - If you have MCP servers set up in Claude Desktop, Claude Code can access them
2. **Three Scope Levels**:
   - Local (project-specific)
   - Project (shared via `.mcp.json`)
   - User (across all projects)

### MCP Server Types Supported:
- **Stdio Servers** (standard input/output)
- **SSE Servers** (Server-Sent Events)
- **HTTP Servers** (REST APIs)

## Setting Up Your Memory & Storage MCP

### Option 1: Create a Unified MCP Server for JARVIS

Create a new MCP server that integrates your mem0, LangChain, and Google Storage:

```python
# mcp_servers/jarvis_memory_storage.py
#!/usr/bin/env python3
"""
JARVIS MCP Server with Memory and Storage Integration
"""

import asyncio
import json
from typing import Dict, Any, List
from mem0 import Memory  # Your mem0 integration
from langchain.memory import ConversationBufferMemory
from google.cloud import storage  # For 30TB storage

class JARVISMemoryStorageMCP:
    def __init__(self):
        # Initialize mem0
        self.memory = Memory.from_config({
            "vector_store": {
                "provider": "your_provider",
                "config": {...}
            }
        })
        
        # Initialize LangChain memory
        self.langchain_memory = ConversationBufferMemory()
        
        # Initialize Google Storage
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket("your-30tb-bucket")
    
    async def handle_tool_call(self, tool_name: str, params: Dict[str, Any]):
        if tool_name == "store_memory":
            return await self.store_memory(params)
        elif tool_name == "retrieve_memory":
            return await self.retrieve_memory(params)
        elif tool_name == "upload_to_storage":
            return await self.upload_to_storage(params)
        elif tool_name == "download_from_storage":
            return await self.download_from_storage(params)
    
    async def store_memory(self, params: Dict[str, Any]):
        """Store information in long-term memory"""
        memory_type = params.get("type", "general")
        content = params["content"]
        metadata = params.get("metadata", {})
        
        # Store in mem0
        self.memory.add(content, metadata=metadata)
        
        # Also store in LangChain for conversation context
        self.langchain_memory.save_context(
            {"input": "Store: " + content},
            {"output": "Stored successfully"}
        )
        
        return {"status": "success", "message": "Memory stored"}
    
    async def retrieve_memory(self, params: Dict[str, Any]):
        """Retrieve from long-term memory"""
        query = params["query"]
        limit = params.get("limit", 10)
        
        # Search mem0
        results = self.memory.search(query, limit=limit)
        
        return {
            "status": "success",
            "memories": results,
            "count": len(results)
        }
    
    async def upload_to_storage(self, params: Dict[str, Any]):
        """Upload to Google Storage"""
        file_path = params["file_path"]
        blob_name = params.get("blob_name", file_path.split("/")[-1])
        
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        
        return {
            "status": "success",
            "blob_name": blob_name,
            "size": blob.size,
            "url": blob.public_url
        }
```

### Option 2: Configure Existing MCP Servers

If you already have MCP servers for mem0 and storage, add them to Claude Code:

```bash
# Add memory MCP server
claude mcp add memory-server /path/to/mem0-mcp-server

# Add storage MCP server  
claude mcp add storage-server /path/to/storage-mcp-server
```

### Option 3: Create Project-Specific Configuration

Create `.mcp.json` in your JARVIS project:

```json
{
  "version": "1.0",
  "servers": {
    "jarvis-memory": {
      "command": "python",
      "args": ["mcp_servers/jarvis_memory_storage.py"],
      "env": {
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/credentials.json"
      }
    }
  }
}
```

## Using MCP in JARVIS

Once configured, you can access MCP features in JARVIS:

```python
# core/mcp_integration.py
class MCPIntegration:
    async def store_conversation(self, user_input: str, jarvis_response: str):
        """Store conversation in long-term memory via MCP"""
        # This will use your MCP server
        await self.mcp_client.call_tool(
            "store_memory",
            {
                "content": f"User: {user_input}\nJARVIS: {jarvis_response}",
                "type": "conversation",
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.session_id
                }
            }
        )
    
    async def search_knowledge(self, query: str):
        """Search long-term memory"""
        return await self.mcp_client.call_tool(
            "retrieve_memory",
            {"query": query, "limit": 20}
        )
```

## Important Notes

1. **Security Warning**: The docs warn to "use third party MCP servers at your own risk" due to potential prompt injection risks

2. **Authentication**: MCP supports OAuth 2.0 for remote servers if needed

3. **Resource Access**: You can reference MCP resources using "@" mentions in Claude

4. **Slash Commands**: MCP server prompts can be exposed as slash commands

## Next Steps

1. **Check Existing MCP Servers**: Run `claude mcp list` to see what's already configured

2. **Create JARVIS MCP Server**: Implement the unified MCP server above

3. **Test Integration**: Use the MCP tools in your JARVIS conversations

4. **Monitor Usage**: Track memory and storage usage through MCP

## Benefits for JARVIS

With MCP integration, JARVIS gains:
- **Persistent Memory**: Remember conversations across sessions
- **Knowledge Base**: Access to 30TB of stored information
- **Context Awareness**: Retrieve relevant past interactions
- **Learning Capability**: Store and refine knowledge over time

This makes JARVIS truly intelligent with long-term memory and massive storage capabilities!