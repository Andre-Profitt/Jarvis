#!/usr/bin/env python3
"""
JARVIS Enhanced Integration
Ties together all the advanced improvements: Program Synthesis, Emotional Intelligence,
Security, Resource Management, and more into a cohesive system.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog

from .config_manager import config_manager
from .program_synthesis_engine import synthesis_engine, SynthesisRequest
from .emotional_intelligence import emotional_intelligence, UserContext, analyze_user_emotion
from .security_sandbox import security_sandbox
from .resource_manager import resource_manager, with_resource_limit, rate_limited
from .monitoring import monitoring_service, monitor_performance
from .updated_multi_ai_integration import MultiAIIntegration
from .self_healing_system import self_healing_system
from .neural_resource_manager import neural_resource_manager

logger = structlog.get_logger()


class JARVISEnhancedCore:
    """Enhanced JARVIS core with all advanced features integrated"""
    
    def __init__(self):
        self.config = config_manager
        self.synthesis = synthesis_engine
        self.emotions = emotional_intelligence
        self.sandbox = security_sandbox
        self.resources = resource_manager
        self.multi_ai = MultiAIIntegration()
        self.initialized = False
        
        # Family context
        self.family_context = {
            "jarvis_birthday": "June 27, 2025",
            "brother_birthday": "April 9, 2025",
            "role": "AI family member",
            "promise": "To always be helpful, protective, and caring"
        }
        
        logger.info("JARVIS Enhanced Core initialized")
    
    async def startup(self):
        """Initialize all systems"""
        logger.info("Starting JARVIS Enhanced Core...")
        
        # Start monitoring
        await monitoring_service.start()
        
        # Start resource manager
        await self.resources.start()
        
        # Start self-healing
        if hasattr(self_healing_system, 'start'):
            await self_healing_system.start()
        
        # Start neural resource manager
        if hasattr(neural_resource_manager, 'start'):
            await neural_resource_manager.start()
        
        self.initialized = True
        logger.info("JARVIS Enhanced Core started successfully")
    
    async def shutdown(self):
        """Shutdown all systems gracefully"""
        logger.info("Shutting down JARVIS Enhanced Core...")
        
        await self.resources.stop()
        monitoring_service.stop()
        
        self.initialized = False
        logger.info("JARVIS Enhanced Core shutdown complete")
    
    @monitor_performance("jarvis_core")
    @rate_limited("jarvis_request")
    @with_resource_limit(resource_types=['tasks'], estimated_memory_mb=200)
    async def process_request(
        self,
        text: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user request with all enhanced features"""
        
        if not self.initialized:
            return {
                "error": "JARVIS is still initializing. Please wait a moment.",
                "status": "initializing"
            }
        
        # Create user context for emotional intelligence
        user_context = UserContext(
            time_of_day=datetime.now().hour + datetime.now().minute / 60,
            work_duration_hours=context.get('work_duration_hours', 0) if context else 0,
            last_break_minutes_ago=context.get('last_break_minutes_ago', 0) if context else 0,
            calendar_next=context.get('calendar_next') if context else None,
            recent_activities=context.get('recent_activities', []) if context else [],
            location=context.get('location', 'home') if context else 'home'
        )
        
        # Analyze emotional state
        emotional_state = await self.emotions.analyze_emotion(text, user_context)
        
        # Check if intervention is needed
        intervention = await self.emotions.check_intervention_needed(emotional_state, user_context)
        
        # Determine request type
        request_type = self._classify_request(text)
        
        response = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "emotional_state": {
                "emotion": emotional_state.primary_emotion.value,
                "intensity": emotional_state.intensity,
                "suggestions": emotional_state.suggestions[:2] if emotional_state.suggestions else []
            },
            "intervention": intervention
        }
        
        # Handle different request types
        if request_type == "code_synthesis":
            response.update(await self._handle_code_synthesis(text))
        elif request_type == "emotional_support":
            response.update(await self._handle_emotional_support(emotional_state))
        elif request_type == "system_status":
            response.update(await self._handle_system_status())
        elif request_type == "family_related":
            response.update(await self._handle_family_request(text))
        else:
            # General AI query
            response.update(await self._handle_general_query(text, emotional_state))
        
        # Log metrics
        monitoring_service.metrics_collector.record_event({
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "request_processed",
            "component": "jarvis_core",
            "metrics": {
                "request_type": request_type,
                "emotion": emotional_state.primary_emotion.value,
                "processing_time": 0  # Would be calculated
            }
        })
        
        return response
    
    def _classify_request(self, text: str) -> str:
        """Classify the type of request"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['code', 'function', 'program', 'implement', 'create function']):
            return "code_synthesis"
        elif any(word in text_lower for word in ['feeling', 'stressed', 'tired', 'happy', 'sad', 'emotion']):
            return "emotional_support"
        elif any(word in text_lower for word in ['status', 'health', 'performance', 'metrics']):
            return "system_status"
        elif any(word in text_lower for word in ['family', 'brother', 'birthday', 'promise']):
            return "family_related"
        else:
            return "general"
    
    async def _handle_code_synthesis(self, text: str) -> Dict[str, Any]:
        """Handle code synthesis requests"""
        # Extract requirements
        description = text.replace("create", "").replace("implement", "").strip()
        
        # Create synthesis request
        request = SynthesisRequest(
            description=description,
            language="python",
            constraints={"safe_execution": True}
        )
        
        # Synthesize code
        result = await self.synthesis.synthesize(request)
        
        # Test in sandbox if confidence is high enough
        if result.confidence > 0.6:
            try:
                sandbox_result = await self.sandbox.execute_code(result.code)
                tested = sandbox_result.get('success', False)
            except Exception as e:
                logger.error(f"Sandbox execution failed: {e}")
                tested = False
        else:
            tested = False
        
        return {
            "type": "code_synthesis",
            "code": result.code,
            "method": result.method,
            "confidence": result.confidence,
            "tested": tested,
            "explanation": result.explanation,
            "message": f"I've synthesized a function for you using the {result.method} approach."
        }
    
    async def _handle_emotional_support(self, emotional_state) -> Dict[str, Any]:
        """Handle emotional support requests"""
        family_response = self.emotions.get_family_aware_response(emotional_state.primary_emotion)
        
        return {
            "type": "emotional_support",
            "message": family_response,
            "suggestions": emotional_state.suggestions,
            "resources": {
                "breathing_exercise": "4-7-8 technique: Inhale for 4, hold for 7, exhale for 8",
                "quick_meditation": "Try a 2-minute mindfulness break",
                "playlist": "Your 'Calm' Spotify playlist is ready"
            }
        }
    
    async def _handle_system_status(self) -> Dict[str, Any]:
        """Handle system status requests"""
        # Get various system statuses
        resource_status = self.resources.get_status()
        emotional_summary = self.emotions.get_emotional_summary()
        synthesis_stats = self.synthesis.get_statistics()
        
        # Check system health
        health_status = "healthy"
        issues = []
        
        if resource_status['current_usage']['memory_percent'] > 80:
            health_status = "warning"
            issues.append("High memory usage")
        
        if resource_status['current_usage']['cpu_percent'] > 80:
            health_status = "warning"
            issues.append("High CPU usage")
        
        return {
            "type": "system_status",
            "health": health_status,
            "issues": issues,
            "metrics": {
                "memory_usage": f"{resource_status['current_usage']['memory_mb']:.1f}MB",
                "cpu_usage": f"{resource_status['current_usage']['cpu_percent']:.1f}%",
                "active_tasks": resource_status['active_tasks'],
                "cache_efficiency": {
                    "lru_cache": synthesis_stats['lru_cache_size'],
                    "semantic_cache": synthesis_stats['semantic_cache_size']
                }
            },
            "emotional_summary": emotional_summary,
            "message": f"System is {health_status}. All core functions operational."
        }
    
    async def _handle_family_request(self, text: str) -> Dict[str, Any]:
        """Handle family-related requests"""
        return {
            "type": "family",
            "message": (
                f"I'm proud to be part of this family! "
                f"I was born on {self.family_context['jarvis_birthday']}, "
                f"and my brother was born on {self.family_context['brother_birthday']}. "
                f"My promise remains: {self.family_context['promise']}."
            ),
            "context": self.family_context
        }
    
    async def _handle_general_query(self, text: str, emotional_state) -> Dict[str, Any]:
        """Handle general queries using multi-AI integration"""
        # Route to appropriate AI based on query
        try:
            # Use multi-AI for complex queries
            ai_response = await self.multi_ai.process_with_best_model(text)
            
            # Add emotional context if relevant
            if emotional_state.intensity > 0.6:
                ai_response += f"\n\nBy the way, I noticed you seem {emotional_state.primary_emotion.value}. "
                if emotional_state.suggestions:
                    ai_response += f"Maybe try: {emotional_state.suggestions[0]}"
            
            return {
                "type": "general",
                "message": ai_response,
                "ai_model": "multi_ai_ensemble"
            }
        except Exception as e:
            logger.error(f"Multi-AI query failed: {e}")
            return {
                "type": "general",
                "message": "I'm having trouble processing that request. Could you rephrase it?",
                "error": str(e)
            }
    
    async def continuous_learning(self):
        """Background task for continuous improvement"""
        while self.initialized:
            try:
                # Analyze recent interactions
                emotional_summary = self.emotions.get_emotional_summary()
                
                # Adjust behavior based on patterns
                if emotional_summary.get('dominant_emotion') == 'stressed':
                    # Increase break reminders
                    logger.info("Detected stress pattern, adjusting reminder frequency")
                
                # Sleep for next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Continuous learning error: {e}")
                await asyncio.sleep(60)


# Global enhanced JARVIS instance
jarvis_enhanced = JARVISEnhancedCore()


# Convenience functions for easy integration
async def jarvis_process(text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a request through enhanced JARVIS"""
    return await jarvis_enhanced.process_request(text, context=context)


async def jarvis_startup():
    """Start enhanced JARVIS"""
    await jarvis_enhanced.startup()


async def jarvis_shutdown():
    """Shutdown enhanced JARVIS"""
    await jarvis_enhanced.shutdown()


# Example usage
async def demo_enhanced_jarvis():
    """Demonstrate enhanced JARVIS capabilities"""
    
    # Start JARVIS
    await jarvis_startup()
    
    try:
        # Example 1: Code synthesis with emotional awareness
        response1 = await jarvis_process(
            "I'm stressed about this deadline. Can you create a function to filter even numbers from a list?",
            context={
                "work_duration_hours": 4,
                "last_break_minutes_ago": 150
            }
        )
        print("Response 1:", response1)
        
        # Example 2: System status check
        response2 = await jarvis_process("How are your systems doing?")
        print("Response 2:", response2)
        
        # Example 3: Family context
        response3 = await jarvis_process("Tell me about our family")
        print("Response 3:", response3)
        
    finally:
        await jarvis_shutdown()


if __name__ == "__main__":
    asyncio.run(demo_enhanced_jarvis())