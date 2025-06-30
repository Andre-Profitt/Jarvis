import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.minimal_jarvis import MinimalJARVIS

@pytest.mark.asyncio
async def test_jarvis_initialization():
    """Test JARVIS can initialize"""
    jarvis = MinimalJARVIS()
    await jarvis.initialize()
    assert jarvis.active == True
    await jarvis.shutdown()

@pytest.mark.asyncio
async def test_jarvis_chat():
    """Test JARVIS can respond to chat"""
    jarvis = MinimalJARVIS()
    await jarvis.initialize()
    
    response = await jarvis.chat("hello")
    assert response is not None
    assert "JARVIS" in response
    
    await jarvis.shutdown()

def test_imports():
    """Test core imports work"""
    try:
        import core.minimal_jarvis
        assert True
    except ImportError:
        assert False, "Failed to import minimal_jarvis"
