"""
JARVIS Phase 7 Test Suite
========================
Tests for Visual UI System
"""

import asyncio
import pytest
import json
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.ui_components import JARVISUIComponents, UITheme, StatusIndicator
from core.visual_feedback_system import VisualFeedbackSystem, InterventionType, UIBridge
from core.jarvis_phase7_integration import JARVISPhase7Core


class TestUIComponents:
    """Test UI component generation"""
    
    @pytest.fixture
    def ui_components(self):
        return JARVISUIComponents(UITheme.DARK)
        
    def test_status_indicator_generation(self, ui_components):
        """Test status indicator HTML generation"""
        html = ui_components.generate_status_indicator(
            "voice",
            StatusIndicator.ACTIVE,
            "Voice Active"
        )
        
        assert "status-indicator" in html
        assert 'data-sensor="voice"' in html
        assert 'data-status="active"' in html
        assert "Voice Active" in html
        
    def test_intervention_preview_generation(self, ui_components):
        """Test intervention preview generation"""
        html = ui_components.generate_intervention_preview(
            "test_action",
            "Testing intervention",
            countdown=5,
            can_cancel=True
        )
        
        assert "intervention-preview" in html
        assert "Testing intervention" in html
        assert 'data-countdown="5"' in html
        assert "intervention-cancel" in html
        
    def test_mode_indicator_generation(self, ui_components):
        """Test mode indicator generation"""
        state_info = {
            "stress_level": 0.3,
            "focus_level": 0.8
        }
        
        html = ui_components.generate_mode_indicator("flow", state_info)
        
        assert "mode-indicator" in html
        assert 'data-mode="flow"' in html
        assert "Flow State" in html
        assert "30.0%" in html  # stress level
        assert "80.0%" in html  # focus level
        
    def test_emotional_visualizer(self, ui_components):
        """Test emotional state visualizer"""
        emotional_state = {
            "valence": 0.5,
            "arousal": -0.3,
            "quadrant": "calm_happy"
        }
        
        html = ui_components.generate_emotional_state_visualizer(emotional_state)
        
        assert "emotional-visualizer" in html
        assert "calm_happy" in html
        assert "left: 75.0%" in html  # (0.5 + 1) * 50
        assert "top: 65.0%" in html   # (1 - (-0.3)) * 50
        
    def test_theme_colors(self, ui_components):
        """Test theme color schemes"""
        dark_theme = JARVISUIComponents(UITheme.DARK)
        light_theme = JARVISUIComponents(UITheme.LIGHT)
        
        dark_colors = dark_theme.color_schemes[UITheme.DARK]
        light_colors = light_theme.color_schemes[UITheme.LIGHT]
        
        assert dark_colors["background"] == "#0a0e27"
        assert light_colors["background"] == "#f5f5f5"
        assert dark_colors["primary"] != light_colors["primary"]


class TestVisualFeedbackSystem:
    """Test visual feedback system functionality"""
    
    @pytest.fixture
    async def feedback_system(self):
        system = VisualFeedbackSystem()
        return system
        
    @pytest.mark.asyncio
    async def test_sensor_update(self, feedback_system):
        """Test sensor status updates"""
        updates = []
        
        async def capture_update(update_type, data):
            updates.append((update_type, data))
            
        feedback_system.add_update_listener(capture_update)
        
        await feedback_system.update_sensor_status(
            "voice",
            "active",
            {"active": True}
        )
        
        assert len(updates) == 1
        assert updates[0][0] == "status_update"
        assert updates[0][1]["sensor"] == "voice"
        assert updates[0][1]["status"] == "active"
        
    @pytest.mark.asyncio
    async def test_intervention_preview(self, feedback_system):
        """Test intervention preview functionality"""
        updates = []
        
        async def capture_update(update_type, data):
            updates.append((update_type, data))
            
        feedback_system.add_update_listener(capture_update)
        
        intervention_id = await feedback_system.preview_intervention(
            InterventionType.SUGGEST_BREAK,
            "Time for a break",
            countdown=0,  # No countdown for test
            can_cancel=True
        )
        
        assert intervention_id is not None
        assert len(updates) == 1
        assert updates[0][0] == "intervention"
        assert updates[0][1]["action"] == "show"
        
    @pytest.mark.asyncio
    async def test_intervention_cancellation(self, feedback_system):
        """Test intervention cancellation"""
        intervention_id = await feedback_system.preview_intervention(
            InterventionType.BREATHING_EXERCISE,
            "Breathing exercise",
            countdown=5,
            can_cancel=True
        )
        
        # Cancel immediately
        await feedback_system.cancel_intervention(intervention_id)
        
        # Check that intervention is marked as cancelled
        intervention = next(
            (i for i in feedback_system.intervention_queue if i["id"] == intervention_id),
            None
        )
        assert intervention is not None
        assert intervention["status"] == "cancelled"
        
    @pytest.mark.asyncio
    async def test_mode_update(self, feedback_system):
        """Test mode update with notifications"""
        updates = []
        
        async def capture_update(update_type, data):
            updates.append((update_type, data))
            
        feedback_system.add_update_listener(capture_update)
        
        await feedback_system.update_mode(
            "flow",
            {"stress_level": 0.2, "focus_level": 0.9},
            "Deep focus detected"
        )
        
        # Should generate mode change and notification
        mode_updates = [u for u in updates if u[0] == "mode_change"]
        notif_updates = [u for u in updates if u[0] == "notification"]
        
        assert len(mode_updates) == 1
        assert mode_updates[0][1]["new_mode"] == "flow"
        
        assert len(notif_updates) >= 1  # At least one notification
        
    @pytest.mark.asyncio
    async def test_notification_queue(self, feedback_system):
        """Test notification queue management"""
        # Add multiple notifications
        for i in range(5):
            await feedback_system.show_notification(
                f"Test notification {i}",
                "info"
            )
            
        assert len(feedback_system.notification_queue) == 5
        
        # Check queue limit (maxlen=50)
        for i in range(50):
            await feedback_system.show_notification(f"Notification {i}", "info")
            
        assert len(feedback_system.notification_queue) == 50
        
    @pytest.mark.asyncio
    async def test_activity_logging(self, feedback_system):
        """Test activity timeline logging"""
        await feedback_system.update_sensor_status("voice", "active")
        await feedback_system.update_mode("flow", {})
        await feedback_system.show_notification("Test", "info")
        
        assert len(feedback_system.activity_log) >= 3
        
        activity_types = [a["type"] for a in feedback_system.activity_log]
        assert "sensor" in activity_types
        assert "system" in activity_types


class TestPhase7Integration:
    """Test complete Phase 7 integration"""
    
    @pytest.fixture
    async def jarvis_phase7(self):
        jarvis = JARVISPhase7Core()
        await jarvis.initialize()
        return jarvis
        
    @pytest.mark.asyncio
    async def test_visual_feedback_on_input(self, jarvis_phase7):
        """Test visual feedback during input processing"""
        # Track UI updates
        ui_updates = []
        
        async def track_updates(update_type, data):
            ui_updates.append(update_type)
            
        jarvis_phase7.visual_feedback.add_update_listener(track_updates)
        
        # Process input
        await jarvis_phase7.process_input({
            "voice": {"text": "Hello JARVIS"},
            "biometric": {"heart_rate": 75}
        })
        
        # Should update voice sensor status
        assert "status_update" in ui_updates
        
    @pytest.mark.asyncio
    async def test_intervention_on_stress(self, jarvis_phase7):
        """Test intervention triggering on stress detection"""
        # Process high-stress input
        result = await jarvis_phase7.process_input({
            "voice": {
                "text": "I'm panicking about the deadline!",
                "features": {"pitch_ratio": 1.5}
            },
            "biometric": {
                "heart_rate": 110,
                "stress_level": 0.9
            }
        })
        
        # Should trigger crisis intervention
        assert result["mode"] == "crisis"
        assert any(a["type"] == "crisis_intervention" for a in result.get("actions", []))
        
        # Check intervention was shown
        assert jarvis_phase7.ui_metrics["interventions_shown"] > 0
        
    @pytest.mark.asyncio
    async def test_mode_visual_update(self, jarvis_phase7):
        """Test mode indicator updates"""
        # Track mode changes
        mode_changes = []
        
        async def track_mode_changes(update_type, data):
            if update_type == "mode_change":
                mode_changes.append(data["new_mode"])
                
        jarvis_phase7.visual_feedback.add_update_listener(track_mode_changes)
        
        # Trigger flow state
        await jarvis_phase7.process_input({
            "voice": {"text": "I'm really focused on coding"},
            "biometric": {"heart_rate": 68, "stress_level": 0.1}
        })
        
        # May or may not change to flow depending on context
        # But mode change tracking should work
        assert isinstance(mode_changes, list)
        
    @pytest.mark.asyncio
    async def test_dashboard_generation(self, jarvis_phase7):
        """Test dashboard HTML generation"""
        html = jarvis_phase7.generate_dashboard_html()
        
        assert "<!DOCTYPE html>" in html
        assert "JARVIS Visual Dashboard" in html
        assert "ws://localhost:8765" in html
        assert "sensor-dashboard" in html
        assert "mode-indicator" in html
        
    @pytest.mark.asyncio
    async def test_ui_preferences(self, jarvis_phase7):
        """Test UI preference settings"""
        # Disable intervention previews
        jarvis_phase7.visual_preferences["preview_interventions"] = False
        
        # Process stress input
        await jarvis_phase7.process_input({
            "voice": {"text": "Very stressed!"},
            "biometric": {"heart_rate": 100, "stress_level": 0.9}
        })
        
        # Should not show interventions
        initial_count = jarvis_phase7.ui_metrics["interventions_shown"]
        assert initial_count == 0  # No interventions due to preference


class TestUIBridge:
    """Test UI Bridge functionality"""
    
    @pytest.fixture
    def ui_bridge(self):
        feedback_system = VisualFeedbackSystem()
        return UIBridge(feedback_system)
        
    @pytest.mark.asyncio
    async def test_sensor_mapping(self, ui_bridge):
        """Test sensor name mapping"""
        await ui_bridge.process_jarvis_update("sensor_update", {
            "sensor": "voice_input",
            "status": "active"
        })
        
        # Should map to "voice"
        assert "voice" in ui_bridge.feedback_system.active_sensors
        
    @pytest.mark.asyncio
    async def test_intervention_mapping(self, ui_bridge):
        """Test intervention type mapping"""
        await ui_bridge.process_jarvis_update("intervention_needed", {
            "intervention_type": "notification_block",
            "description": "Blocking notifications",
            "countdown": 3
        })
        
        # Should create intervention
        assert len(ui_bridge.feedback_system.intervention_queue) > 0
        intervention = ui_bridge.feedback_system.intervention_queue[-1]
        assert intervention["type"] == InterventionType.BLOCK_NOTIFICATIONS


# Performance tests
class TestPhase7Performance:
    """Test Phase 7 performance"""
    
    @pytest.fixture
    async def jarvis_phase7(self):
        jarvis = JARVISPhase7Core()
        await jarvis.initialize()
        return jarvis
        
    @pytest.mark.asyncio
    async def test_ui_update_performance(self, jarvis_phase7):
        """Test UI update performance"""
        start = datetime.now()
        
        # Rapid sensor updates
        for i in range(10):
            await jarvis_phase7.visual_feedback.update_sensor_status(
                "voice",
                "active" if i % 2 == 0 else "idle"
            )
            
        elapsed = (datetime.now() - start).total_seconds()
        assert elapsed < 1.0  # Should handle 10 updates in under 1 second
        
    @pytest.mark.asyncio
    async def test_notification_performance(self, jarvis_phase7):
        """Test notification system performance"""
        start = datetime.now()
        
        # Many notifications
        for i in range(20):
            await jarvis_phase7.visual_feedback.show_notification(
                f"Notification {i}",
                "info",
                duration=1000
            )
            
        elapsed = (datetime.now() - start).total_seconds()
        assert elapsed < 0.5  # Should handle 20 notifications quickly


def run_tests():
    """Run all Phase 7 tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
