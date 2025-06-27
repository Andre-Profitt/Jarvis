#!/usr/bin/env python3
"""
Basic tests for JARVIS
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestJARVISCore:
    """Test core JARVIS functionality"""
    
    @pytest.mark.asyncio
    async def test_multi_ai_initialization(self):
        """Test multi-AI integration initialization"""
        
        from core.updated_multi_ai_integration import multi_ai
        
        await multi_ai.initialize()
        assert len(multi_ai.available_models) > 0
    
    def test_imports(self):
        """Test all imports work"""
        
        imports = [
            "from core.real_claude_integration import claude_integration",
            "from core.real_openai_integration import openai_integration",
            "from core.real_elevenlabs_integration import elevenlabs_integration",
            "from core.websocket_security import websocket_security",
            "from core.health_checks import health_checker"
        ]
        
        for import_stmt in imports:
            try:
                exec(import_stmt)
            except ImportError as e:
                pytest.fail(f"Import failed: {import_stmt} - {e}")

if __name__ == "__main__":
    pytest.main([__file__])
