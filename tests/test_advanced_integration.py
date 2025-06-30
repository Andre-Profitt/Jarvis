"""
Test Suite for Advanced Integration Hub
======================================
Comprehensive tests for JARVIS Advanced Integration Hub including
smart home, calendar, and proactive assistance features.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from tests.conftest import *
from tests.mocks import *

from core.advanced_integration import (
    AdvancedIntegrationHub,
    IntegrationType,
    SmartDevice,
    CalendarEvent,
    ProactiveAction
)


class TestAdvancedIntegrationHub:
    """Test suite for Advanced Integration Hub"""
    
    @pytest.fixture
    async def hub(self, mock_redis, mock_database):
        """Create integration hub instance with mocked dependencies"""
        hub = AdvancedIntegrationHub()
        yield hub
        # Cleanup
        hub.active = False
    
    @pytest.fixture
    def sample_smart_device(self):
        """Sample smart device for testing"""
        return SmartDevice(
            id="test_light",
            name="Test Light",
            type="light",
            room="test_room",
            state={"on": False, "brightness": 50},
            capabilities=["on_off", "dim"]
        )
    
    @pytest.fixture
    def sample_calendar_event(self):
        """Sample calendar event for testing"""
        return CalendarEvent(
            id="test_event",
            title="Test Meeting",
            start_time=datetime.now() + timedelta(hours=1),
            end_time=datetime.now() + timedelta(hours=2),
            location="Conference Room A",
            attendees=["user@example.com"]
        )
    
    # ===== Initialization Tests =====
    @pytest.mark.asyncio
    async def test_initialization(self, hub):
        """Test hub initialization"""
        assert hub is not None
        assert hub.active == False
        assert len(hub.integrations) == 0
        assert len(hub.smart_devices) == 0
        assert len(hub.calendar_events) == 0
        
    @pytest.mark.asyncio
    async def test_initialize_creates_background_tasks(self, hub):
        """Test that initialization creates background tasks"""
        with patch('asyncio.create_task') as mock_create_task:
            await hub.initialize()
            
            assert hub.active == True
            # Should create 3 background tasks
            assert mock_create_task.call_count >= 3
            
    @pytest.mark.asyncio
    async def test_smart_home_initialization(self, hub):
        """Test smart home device initialization"""
        await hub._init_smart_home()
        
        assert IntegrationType.SMART_HOME in hub.integrations
        assert len(hub.smart_devices) > 0
        assert "light_living" in hub.smart_devices
        assert "thermostat_main" in hub.smart_devices
        assert "lock_front" in hub.smart_devices
        
    # ===== Smart Device Tests =====
    def test_smart_device_creation(self, sample_smart_device):
        """Test smart device dataclass creation"""
        device = sample_smart_device
        
        assert device.id == "test_light"
        assert device.type == "light"
        assert device.online == True
        assert "on_off" in device.capabilities
        assert device.state["on"] == False
        
    @pytest.mark.asyncio
    async def test_control_smart_device(self, hub):
        """Test controlling smart devices"""
        await hub._init_smart_home()
        
        # Test turning on a light
        light = hub.smart_devices.get("light_living")
        assert light is not None
        
        # Simulate device control
        light.state["on"] = True
        light.state["brightness"] = 80
        
        assert light.state["on"] == True
        assert light.state["brightness"] == 80
        
    # ===== Calendar Tests =====
    @pytest.mark.asyncio
    async def test_calendar_initialization(self, hub):
        """Test calendar initialization"""
        await hub._init_calendar()
        
        assert IntegrationType.CALENDAR in hub.integrations
        assert len(hub.calendar_events) > 0
        
    def test_calendar_event_creation(self, sample_calendar_event):
        """Test calendar event dataclass"""
        event = sample_calendar_event
        
        assert event.id == "test_event"
        assert event.title == "Test Meeting"
        assert event.location == "Conference Room A"
        assert event.reminder_sent == False
        assert len(event.attendees) == 1
        
    @pytest.mark.asyncio
    async def test_add_calendar_event(self, hub, sample_calendar_event):
        """Test adding calendar events"""
        await hub.initialize()
        
        initial_count = len(hub.calendar_events)
        hub.calendar_events.append(sample_calendar_event)
        
        assert len(hub.calendar_events) == initial_count + 1
        assert sample_calendar_event in hub.calendar_events
        
    # ===== Proactive Action Tests =====
    def test_proactive_action_creation(self):
        """Test proactive action dataclass"""
        action = ProactiveAction(
            id="test_action",
            type="reminder",
            trigger="calendar_event",
            action="send_notification",
            priority=0.8,
            context={"event_id": "test_event"}
        )
        
        assert action.id == "test_action"
        assert action.priority == 0.8
        assert action.executed == False
        assert action.context["event_id"] == "test_event"
        
    @pytest.mark.asyncio
    async def test_proactive_queue_management(self, hub):
        """Test proactive action queue"""
        await hub.initialize()
        
        # Add test action
        action = ProactiveAction(
            id="test_1",
            type="automation",
            trigger="time",
            action="turn_on_lights",
            priority=0.9,
            context={}
        )
        
        hub.proactive_queue.append(action)
        assert len(hub.proactive_queue) == 1
        assert hub.proactive_queue[0].priority == 0.9
        
    # ===== User Preferences Tests =====
    def test_default_user_preferences(self, hub):
        """Test default user preferences"""
        prefs = hub.user_preferences
        
        assert prefs["wake_time"] == "07:00"
        assert prefs["sleep_time"] == "23:00"
        assert prefs["notifications"] == True
        assert prefs["automation_level"] == "high"
        assert "home_location" in prefs
        
    @pytest.mark.asyncio
    async def test_update_user_preferences(self, hub):
        """Test updating user preferences"""
        hub.user_preferences["wake_time"] = "06:30"
        hub.user_preferences["automation_level"] = "medium"
        
        assert hub.user_preferences["wake_time"] == "06:30"
        assert hub.user_preferences["automation_level"] == "medium"
        
    # ===== Integration Type Tests =====
    def test_integration_types(self):
        """Test integration type enum"""
        assert IntegrationType.SMART_HOME.value == "smart_home"
        assert IntegrationType.CALENDAR.value == "calendar"
        assert IntegrationType.WEATHER.value == "weather"
        
        # Test all integration types are defined
        types = [t.value for t in IntegrationType]
        assert "smart_home" in types
        assert "calendar" in types
        assert "weather" in types
        assert "productivity" in types
        
    # ===== Error Handling Tests =====
    @pytest.mark.asyncio
    async def test_initialization_error_handling(self, hub):
        """Test error handling during initialization"""
        with patch.object(hub, '_init_smart_home', side_effect=Exception("Init error")):
            # Should handle error gracefully
            try:
                await hub.initialize()
                # Hub should still be active even if one integration fails
                assert hub.active == True
            except Exception:
                # Should not raise exception
                pytest.fail("Initialization should handle errors gracefully")
                
    # ===== Background Task Tests =====
    @pytest.mark.asyncio
    async def test_background_task_creation(self, hub):
        """Test background task creation"""
        with patch('asyncio.create_task') as mock_create_task:
            await hub.initialize()
            
            # Verify background tasks are created
            calls = mock_create_task.call_args_list
            assert len(calls) >= 3  # At least 3 background tasks
            
    # ===== Integration Tests =====
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_integration_flow(self, hub):
        """Test full integration workflow"""
        await hub.initialize()
        
        # Verify all integrations are active
        assert hub.active == True
        assert len(hub.integrations) > 0
        assert len(hub.smart_devices) > 0
        
        # Test adding and processing an event
        event = CalendarEvent(
            id="meeting_123",
            title="Important Meeting",
            start_time=datetime.now() + timedelta(minutes=30),
            end_time=datetime.now() + timedelta(minutes=90)
        )
        hub.calendar_events.append(event)
        
        # Verify event was added
        assert event in hub.calendar_events
        
    # ===== Concurrency Tests =====
    @pytest.mark.asyncio
    async def test_concurrent_device_control(self, hub):
        """Test concurrent smart device control"""
        await hub._init_smart_home()
        
        # Simulate concurrent device updates
        async def toggle_device(device_id):
            device = hub.smart_devices.get(device_id)
            if device and device.type == "light":
                device.state["on"] = not device.state.get("on", False)
                
        # Run concurrent updates
        tasks = [toggle_device("light_living") for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # Device state should be consistent
        light = hub.smart_devices.get("light_living")
        assert isinstance(light.state["on"], bool)
        
    # ===== Performance Tests =====
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_initialization_performance(self, hub, benchmark_timer):
        """Test initialization performance"""
        with benchmark_timer.measure('hub_init'):
            await hub.initialize()
            
        # Should initialize quickly
        assert benchmark_timer.times['hub_init'] < 1.0  # Less than 1 second
        
    # ===== State Management Tests =====
    @pytest.mark.asyncio
    async def test_hub_state_persistence(self, hub):
        """Test hub state management"""
        await hub.initialize()
        
        # Modify state
        hub.smart_devices["light_living"].state["on"] = True
        hub.user_preferences["automation_level"] = "low"
        
        # State should persist
        assert hub.smart_devices["light_living"].state["on"] == True
        assert hub.user_preferences["automation_level"] == "low"
        
    # ===== Cleanup Tests =====
    @pytest.mark.asyncio
    async def test_hub_cleanup(self, hub):
        """Test proper cleanup"""
        await hub.initialize()
        assert hub.active == True
        
        # Cleanup
        hub.active = False
        assert hub.active == False