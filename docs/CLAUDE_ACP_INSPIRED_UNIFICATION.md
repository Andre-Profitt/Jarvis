# Claude Unification Using ACP-Inspired Architecture

## Overview

Based on IBM's Agent Communication Protocol (ACP), we can create a unified communication system between Claude Desktop and Claude Code that treats each as an independent agent with well-defined capabilities.

## ACP-Inspired Architecture for Claude Unification

### 1. **Agent Identity and Capabilities**

Each Claude instance advertises its capabilities:

```json
// claude_desktop_agent.json
{
  "agent_id": "claude-desktop-001",
  "name": "Claude Desktop Designer",
  "version": "1.0.0",
  "capabilities": [
    {
      "name": "design_system",
      "description": "Create system architecture designs",
      "input_types": ["text", "requirements"],
      "output_types": ["artifact", "diagram", "documentation"]
    },
    {
      "name": "prototype_code",
      "description": "Generate code prototypes and examples",
      "input_types": ["text", "design_spec"],
      "output_types": ["code", "artifact"]
    }
  ],
  "endpoints": {
    "task": "/tasks",
    "status": "/status",
    "results": "/results/{task_id}"
  }
}

// claude_code_agent.json
{
  "agent_id": "claude-code-001",
  "name": "Claude Code Implementer",
  "version": "1.0.0",
  "capabilities": [
    {
      "name": "implement_feature",
      "description": "Implement features in codebase",
      "input_types": ["design_spec", "artifact", "file_path"],
      "output_types": ["code_files", "test_files", "commit"]
    },
    {
      "name": "refactor_code",
      "description": "Refactor and optimize existing code",
      "input_types": ["file_path", "requirements"],
      "output_types": ["code_files", "performance_report"]
    }
  ]
}
```

### 2. **Standardized Message Format**

Following ACP's multipart message structure:

```python
# core/claude_communication_protocol.py
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json
import uuid

class MessagePartType(Enum):
    TEXT = "text/plain"
    JSON = "application/json"
    CODE = "text/x-python"
    ARTIFACT = "application/x-claude-artifact"
    IMAGE = "image/png"
    REFERENCE = "application/x-reference"

class MessagePart:
    def __init__(self, content_type: MessagePartType, content: Any, 
                 name: Optional[str] = None, content_url: Optional[str] = None):
        self.content_type = content_type
        self.content = content
        self.name = name  # For semantic tagging (e.g., "design_doc", "api_spec")
        self.content_url = content_url  # For large content stored elsewhere

class ClaudeMessage:
    def __init__(self, from_agent: str, to_agent: str, task_type: str):
        self.message_id = str(uuid.uuid4())
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.task_type = task_type
        self.timestamp = datetime.now().isoformat()
        self.parts: List[MessagePart] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_part(self, part: MessagePart):
        self.parts.append(part)
    
    def to_json(self) -> str:
        return json.dumps({
            "message_id": self.message_id,
            "from": self.from_agent,
            "to": self.to_agent,
            "task_type": self.task_type,
            "timestamp": self.timestamp,
            "parts": [
                {
                    "content_type": part.content_type.value,
                    "content": part.content,
                    "name": part.name,
                    "content_url": part.content_url
                }
                for part in self.parts
            ],
            "metadata": self.metadata
        })
```

### 3. **Task Request Structure**

Structured task delegation between Claudes:

```python
class TaskRequest:
    def __init__(self, task_id: str, task_type: str, priority: int = 1):
        self.task_id = task_id
        self.task_type = task_type
        self.priority = priority
        self.status = "pending"
        self.created_at = datetime.now()
        self.message: Optional[ClaudeMessage] = None
        self.response: Optional[ClaudeMessage] = None
    
    def create_design_to_implementation_task(self, design_artifact: str, 
                                           requirements: str) -> ClaudeMessage:
        """Create a task to implement a design from Claude Desktop"""
        msg = ClaudeMessage(
            from_agent="claude-desktop",
            to_agent="claude-code",
            task_type="implement_feature"
        )
        
        # Add design artifact
        msg.add_part(MessagePart(
            content_type=MessagePartType.ARTIFACT,
            content=design_artifact,
            name="design_specification"
        ))
        
        # Add requirements
        msg.add_part(MessagePart(
            content_type=MessagePartType.TEXT,
            content=requirements,
            name="implementation_requirements"
        ))
        
        # Add context from memory
        msg.add_part(MessagePart(
            content_type=MessagePartType.REFERENCE,
            content_url="mcp://jarvis-memory/context/current-project",
            name="project_context"
        ))
        
        self.message = msg
        return msg
```

### 4. **Communication Hub**

Central broker for Claude-to-Claude communication:

```python
# core/claude_communication_hub.py
import asyncio
from typing import Dict, List, Optional
import aiofiles
from pathlib import Path

class ClaudeCommunicationHub:
    """
    ACP-inspired communication hub for Claude agents
    """
    def __init__(self):
        self.agents: Dict[str, Dict] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.handoff_dir = Path("handoff")
        self.handoff_dir.mkdir(exist_ok=True)
    
    async def register_agent(self, agent_config: Dict):
        """Register a Claude agent with its capabilities"""
        agent_id = agent_config["agent_id"]
        self.agents[agent_id] = agent_config
        print(f"Registered agent: {agent_id}")
    
    async def submit_task(self, task: TaskRequest):
        """Submit a task for processing"""
        await self.task_queue.put(task)
        self.active_tasks[task.task_id] = task
        
        # Persist to filesystem for manual handoff
        await self._persist_task(task)
    
    async def _persist_task(self, task: TaskRequest):
        """Save task to filesystem for manual processing"""
        task_file = self.handoff_dir / f"{task.task_id}.json"
        
        task_data = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "message": json.loads(task.message.to_json()) if task.message else None
        }
        
        async with aiofiles.open(task_file, 'w') as f:
            await f.write(json.dumps(task_data, indent=2))
    
    async def get_pending_tasks(self, agent_id: str) -> List[TaskRequest]:
        """Get pending tasks for a specific agent"""
        pending = []
        
        # Check filesystem for tasks
        for task_file in self.handoff_dir.glob("*.json"):
            async with aiofiles.open(task_file, 'r') as f:
                data = json.loads(await f.read())
                
            if data.get("status") == "pending":
                message_data = data.get("message", {})
                if message_data.get("to") == agent_id:
                    # Reconstruct task
                    task = TaskRequest(
                        task_id=data["task_id"],
                        task_type=data["task_type"]
                    )
                    task.status = data["status"]
                    pending.append(task)
        
        return pending
    
    async def complete_task(self, task_id: str, response: ClaudeMessage):
        """Mark a task as complete with response"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.response = response
            task.status = "completed"
            
            # Update persisted task
            await self._persist_task(task)
```

### 5. **Practical Implementation**

Create helper scripts for the workflow:

```python
# scripts/claude_handoff.py
#!/usr/bin/env python3
"""
Facilitate handoff between Claude Desktop and Claude Code
using ACP-inspired protocol
"""

import asyncio
import json
from pathlib import Path
from claude_communication_protocol import ClaudeMessage, MessagePart, MessagePartType
from claude_communication_hub import ClaudeCommunicationHub

class ClaudeHandoffManager:
    def __init__(self):
        self.hub = ClaudeCommunicationHub()
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
    
    async def create_implementation_handoff(
        self, 
        feature_name: str,
        design_file: str,
        requirements: str,
        context: Dict[str, Any]
    ):
        """Create a handoff from Claude Desktop to Claude Code"""
        
        # Create task
        task_id = f"impl_{feature_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = TaskRequest(task_id, "implement_feature")
        
        # Build message
        msg = ClaudeMessage(
            from_agent="claude-desktop",
            to_agent="claude-code",
            task_type="implement_feature"
        )
        
        # Add design artifact
        with open(design_file, 'r') as f:
            design_content = f.read()
        
        msg.add_part(MessagePart(
            content_type=MessagePartType.ARTIFACT,
            content=design_content,
            name="design"
        ))
        
        # Add requirements
        msg.add_part(MessagePart(
            content_type=MessagePartType.TEXT,
            content=requirements,
            name="requirements"
        ))
        
        # Add context
        msg.add_part(MessagePart(
            content_type=MessagePartType.JSON,
            content=context,
            name="context"
        ))
        
        # Add memory reference
        msg.add_part(MessagePart(
            content_type=MessagePartType.REFERENCE,
            content_url=f"mcp://jarvis-memory/project/{feature_name}",
            name="memory_context"
        ))
        
        task.message = msg
        
        # Submit task
        await self.hub.submit_task(task)
        
        print(f"✅ Handoff created: {task_id}")
        print(f"📁 Saved to: handoff/{task_id}.json")
        print(f"🎯 For Claude Code: Check pending tasks")
        
        return task_id
    
    async def check_pending_implementations(self):
        """Check for pending implementation tasks"""
        pending = await self.hub.get_pending_tasks("claude-code")
        
        if not pending:
            print("No pending implementation tasks")
            return
        
        print(f"Found {len(pending)} pending tasks:")
        for task in pending:
            print(f"- {task.task_id}: {task.task_type}")
            print(f"  Created: {task.created_at}")
            print(f"  Status: {task.status}")
    
    async def complete_implementation(self, task_id: str, 
                                    implemented_files: List[str],
                                    summary: str):
        """Mark an implementation as complete"""
        
        # Create response message
        response = ClaudeMessage(
            from_agent="claude-code",
            to_agent="claude-desktop",
            task_type="implementation_complete"
        )
        
        # Add summary
        response.add_part(MessagePart(
            content_type=MessagePartType.TEXT,
            content=summary,
            name="implementation_summary"
        ))
        
        # Add file list
        response.add_part(MessagePart(
            content_type=MessagePartType.JSON,
            content={"files": implemented_files},
            name="implemented_files"
        ))
        
        # Complete task
        await self.hub.complete_task(task_id, response)
        
        print(f"✅ Task {task_id} marked as complete")

# CLI Interface
async def main():
    import sys
    
    manager = ClaudeHandoffManager()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  claude_handoff.py create <feature> <design_file> <requirements>")
        print("  claude_handoff.py check")
        print("  claude_handoff.py complete <task_id> <summary>")
        return
    
    command = sys.argv[1]
    
    if command == "create":
        feature = sys.argv[2]
        design_file = sys.argv[3]
        requirements = " ".join(sys.argv[4:])
        
        await manager.create_implementation_handoff(
            feature, design_file, requirements, 
            {"project": "JARVIS", "version": "2.0"}
        )
    
    elif command == "check":
        await manager.check_pending_implementations()
    
    elif command == "complete":
        task_id = sys.argv[2]
        summary = " ".join(sys.argv[3:])
        
        # In real use, would parse actual file list
        await manager.complete_implementation(
            task_id, ["core/new_feature.py"], summary
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### 6. **Integration with MCP**

Since you have MCP for memory, we can extend the protocol:

```python
# Store task context in MCP
msg.add_part(MessagePart(
    content_type=MessagePartType.REFERENCE,
    content_url="mcp://jarvis-memory/store",
    content={
        "operation": "store_context",
        "data": {
            "task_id": task_id,
            "design": design_content,
            "timestamp": datetime.now().isoformat()
        }
    },
    name="persist_to_memory"
))
```

## Benefits of ACP-Inspired Approach

1. **Clear Communication Protocol**: Structured messages with defined types
2. **Capability Discovery**: Each Claude advertises what it can do
3. **Asynchronous by Default**: Tasks can be long-running
4. **Persistence**: All communication is saved for recovery
5. **Extensible**: Easy to add new message types or capabilities
6. **Tool-Agnostic**: Works with filesystem, MCP, or direct API

## Quick Start

1. **In Claude Desktop**: 
   ```
   "Create a design for feature X and prepare handoff to Claude Code"
   ```

2. **In Claude Code**:
   ```
   "Check pending tasks from Claude Desktop and implement"
   ```

3. **Use the CLI**:
   ```bash
   # Create handoff
   python scripts/claude_handoff.py create neural-cache design.md "Implement with Redis backend"
   
   # Check pending
   python scripts/claude_handoff.py check
   
   # Complete task
   python scripts/claude_handoff.py complete impl_neural-cache_20240628_143022 "Implemented Redis cache"
   ```

This ACP-inspired architecture provides a robust foundation for Claude-to-Claude communication!