# Unified Input Pipeline for JARVIS
# Phase 1 Implementation - Production Ready

import asyncio
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
import logging
import uuid
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# INPUT TYPE DETECTION & ROUTING
# ============================================

class InputType(Enum):
    """All possible input types JARVIS can handle"""
    VOICE = auto()
    VISION = auto()
    BIOMETRIC = auto()
    TEXT = auto()
    TEMPORAL = auto()
    SCREEN = auto()
    GESTURE = auto()
    ENVIRONMENTAL = auto()
    SYSTEM = auto()
    UNKNOWN = auto()

class Priority(Enum):
    """Priority levels for input processing"""
    CRITICAL = 0  # Highest priority (emergency)
    URGENT = 1    # High priority (stress, important notifications)
    HIGH = 2      # Above normal (active interactions)
    NORMAL = 3    # Standard priority
    LOW = 4       # Background processing
    DEFERRED = 5  # Can be processed later

@dataclass
class InputMetadata:
    """Metadata for tracking input through the pipeline"""
    input_id: str
    source: str
    timestamp: datetime
    latency_budget: float = 0.1  # Max processing time in seconds
    retry_count: int = 0
    parent_id: Optional[str] = None
    correlation_ids: List[str] = field(default_factory=list)

@dataclass
class UnifiedInput:
    """Unified input container for all modalities"""
    type: InputType
    data: Any
    metadata: InputMetadata
    priority: Priority
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue sorting"""
        return self.priority.value < other.priority.value

class InputDetector:
    """Sophisticated input type detection with confidence scoring"""
    
    def __init__(self):
        self.detection_rules = self._init_detection_rules()
        self.ml_detector = None  # Placeholder for ML model
        
    def _init_detection_rules(self) -> Dict[InputType, List[callable]]:
        """Initialize rule-based detection"""
        return {
            InputType.VOICE: [
                lambda d: 'waveform' in d or 'audio' in d or 'audio_data' in d,
                lambda d: 'sample_rate' in d and 'duration' in d,
                lambda d: isinstance(d.get('data'), np.ndarray) and len(d.get('data', [])) > 1000
            ],
            InputType.VISION: [
                lambda d: 'image' in d or 'frame' in d or 'video' in d,
                lambda d: 'pixels' in d or 'resolution' in d,
                lambda d: isinstance(d.get('data'), np.ndarray) and len(d.get('data', {}).shape) >= 3
            ],
            InputType.BIOMETRIC: [
                lambda d: any(key in d for key in ['heart_rate', 'blood_pressure', 'temperature']),
                lambda d: 'biometric' in d or 'health' in d,
                lambda d: 'stress_level' in d or 'hrv' in d
            ],
            InputType.TEXT: [
                lambda d: 'text' in d or 'message' in d or 'query' in d,
                lambda d: isinstance(d.get('content'), str),
                lambda d: 'chat' in d or 'command' in d
            ],
            InputType.TEMPORAL: [
                lambda d: 'schedule' in d or 'calendar' in d,
                lambda d: 'reminder' in d or 'alarm' in d,
                lambda d: 'timestamp' in d and 'event' in d
            ],
            InputType.SCREEN: [
                lambda d: 'screenshot' in d or 'screen_capture' in d,
                lambda d: 'window_id' in d or 'application' in d
            ],
            InputType.GESTURE: [
                lambda d: 'gesture' in d or 'motion' in d,
                lambda d: 'accelerometer' in d or 'gyroscope' in d
            ],
            InputType.ENVIRONMENTAL: [
                lambda d: any(key in d for key in ['temperature', 'humidity', 'light', 'noise']),
                lambda d: 'environment' in d or 'ambient' in d
            ],
            InputType.SYSTEM: [
                lambda d: 'system' in d or 'performance' in d,
                lambda d: any(key in d for key in ['cpu', 'memory', 'disk', 'network'])
            ]
        }
    
    def detect(self, raw_input: Any) -> Tuple[InputType, float]:
        """
        Detect input type with confidence score
        Returns: (InputType, confidence)
        """
        if not isinstance(raw_input, dict):
            return InputType.UNKNOWN, 0.0
            
        scores = {}
        
        # Apply rule-based detection
        for input_type, rules in self.detection_rules.items():
            matches = sum(1 for rule in rules if self._safe_apply_rule(rule, raw_input))
            confidence = matches / len(rules) if rules else 0
            if confidence > 0:
                scores[input_type] = confidence
        
        # Return best match
        if scores:
            best_type = max(scores, key=scores.get)
            return best_type, scores[best_type]
        
        return InputType.UNKNOWN, 0.0
    
    def _safe_apply_rule(self, rule: callable, data: dict) -> bool:
        """Safely apply detection rule"""
        try:
            return rule(data)
        except:
            return False

class PriorityCalculator:
    """Dynamic priority calculation based on content and context"""
    
    def __init__(self):
        self.critical_thresholds = {
            'heart_rate': (50, 120),  # Below or above = critical
            'stress_level': 0.8,
            'error_rate': 0.5,
            'response_time': 5000,  # ms
        }
        self.context_weights = {
            'user_state': 0.4,
            'time_sensitivity': 0.3,
            'interaction_type': 0.2,
            'historical_importance': 0.1
        }
    
    def calculate(self, input_type: InputType, data: Any, context: Dict[str, Any]) -> Priority:
        """Calculate priority with context awareness"""
        
        # Check for critical conditions first
        if self._is_critical(input_type, data):
            return Priority.CRITICAL
            
        # Time-sensitive checks
        if self._is_time_sensitive(data, context):
            return Priority.URGENT
            
        # Type-based base priority
        base_priority = self._get_base_priority(input_type)
        
        # Adjust based on context
        adjustment = self._calculate_context_adjustment(context)
        
        # Apply adjustment
        final_priority_value = base_priority.value + adjustment
        final_priority_value = max(0, min(5, final_priority_value))  # Clamp
        
        # Convert back to Priority enum
        for priority in Priority:
            if priority.value == round(final_priority_value):
                return priority
                
        return Priority.NORMAL
    
    def _is_critical(self, input_type: InputType, data: Any) -> bool:
        """Check for critical conditions"""
        if input_type == InputType.BIOMETRIC:
            hr = data.get('heart_rate', 70)
            if hr < self.critical_thresholds['heart_rate'][0] or hr > self.critical_thresholds['heart_rate'][1]:
                return True
            if data.get('stress_level', 0) > self.critical_thresholds['stress_level']:
                return True
                
        elif input_type == InputType.SYSTEM:
            if data.get('error_rate', 0) > self.critical_thresholds['error_rate']:
                return True
            if data.get('response_time', 0) > self.critical_thresholds['response_time']:
                return True
                
        elif input_type == InputType.VOICE:
            # Panic detection in voice
            features = data.get('features', {})
            if features.get('pitch_variance', 0) > 0.8 and features.get('volume', 0) > 0.8:
                return True
                
        return False
    
    def _is_time_sensitive(self, data: Any, context: Dict[str, Any]) -> bool:
        """Check for time-sensitive inputs"""
        # Ongoing meetings/calls
        if context.get('active_call') or context.get('in_meeting'):
            return True
            
        # Deadline approaching
        if context.get('deadline_minutes', float('inf')) < 30:
            return True
            
        # Real-time interaction
        if data.get('real_time') or data.get('live_stream'):
            return True
            
        return False
    
    def _get_base_priority(self, input_type: InputType) -> Priority:
        """Get base priority for input type"""
        priority_map = {
            InputType.VOICE: Priority.HIGH,
            InputType.BIOMETRIC: Priority.HIGH,
            InputType.TEXT: Priority.NORMAL,
            InputType.VISION: Priority.NORMAL,
            InputType.SYSTEM: Priority.NORMAL,
            InputType.TEMPORAL: Priority.NORMAL,
            InputType.SCREEN: Priority.LOW,
            InputType.ENVIRONMENTAL: Priority.LOW,
            InputType.GESTURE: Priority.HIGH,
            InputType.UNKNOWN: Priority.LOW
        }
        return priority_map.get(input_type, Priority.NORMAL)
    
    def _calculate_context_adjustment(self, context: Dict[str, Any]) -> float:
        """Calculate priority adjustment from context"""
        adjustment = 0.0
        
        # User state impact
        user_state = context.get('user_state', {})
        if user_state.get('flow_state'):
            adjustment += 1.0  # Lower priority during flow
        elif user_state.get('stressed'):
            adjustment -= 0.5  # Higher priority when stressed
            
        # Activity impact
        activity = context.get('current_activity', '')
        if activity in ['presentation', 'meeting', 'call']:
            adjustment -= 1.0  # Higher priority during important activities
            
        return adjustment

# ============================================
# UNIFIED INPUT PIPELINE
# ============================================

class UnifiedInputPipeline:
    """Main pipeline for processing all inputs"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.input_detector = InputDetector()
        self.priority_calculator = PriorityCalculator()
        self.priority_queue = []
        self.processing_tasks = {}
        self.processors = {}
        self.max_queue_size = max_queue_size
        
        # Metrics
        self.metrics = {
            'total_processed': 0,
            'by_type': {t: 0 for t in InputType},
            'by_priority': {p: 0 for p in Priority},
            'avg_latency': 0.0,
            'queue_size': 0
        }
        
        # Buffer for non-critical inputs
        self.buffer = deque(maxlen=1000)
        
        # Initialize processors
        self._init_processors()
        
        # Start background processing
        self.background_task = None
        
    def _init_processors(self):
        """Initialize modality processors"""
        # Import existing processors if available
        try:
            from .voice_processor import VoiceProcessor
            from .vision_processor import VisionProcessor
            from .biometric_processor import BiometricProcessor
            
            self.processors = {
                InputType.VOICE: VoiceProcessor(),
                InputType.VISION: VisionProcessor(),
                InputType.BIOMETRIC: BiometricProcessor(),
            }
        except ImportError:
            # Use default processors
            self.processors = {
                InputType.VOICE: DefaultProcessor("voice"),
                InputType.VISION: DefaultProcessor("vision"),
                InputType.BIOMETRIC: DefaultProcessor("biometric"),
                InputType.TEXT: DefaultProcessor("text"),
                InputType.TEMPORAL: DefaultProcessor("temporal"),
                InputType.SCREEN: DefaultProcessor("screen"),
                InputType.ENVIRONMENTAL: DefaultProcessor("environmental"),
                InputType.SYSTEM: DefaultProcessor("system"),
                InputType.GESTURE: DefaultProcessor("gesture"),
                InputType.UNKNOWN: DefaultProcessor("unknown")
            }
    
    async def start(self):
        """Start the pipeline"""
        logger.info("Starting Unified Input Pipeline")
        self.background_task = asyncio.create_task(self._process_background())
        
    async def stop(self):
        """Stop the pipeline"""
        logger.info("Stopping Unified Input Pipeline")
        if self.background_task:
            self.background_task.cancel()
            
    async def process_input(self, raw_input: Any, metadata: Optional[Dict] = None) -> Any:
        """
        Main entry point for all inputs
        
        Args:
            raw_input: The input data
            metadata: Optional metadata about the input
            
        Returns:
            Processing result or task ID for async processing
        """
        # Create metadata
        input_metadata = InputMetadata(
            input_id=self._generate_id(),
            source=metadata.get('source', 'unknown') if metadata else 'unknown',
            timestamp=datetime.now()
        )
        
        # Detect input type
        input_type, confidence = self.input_detector.detect(raw_input)
        
        if confidence < 0.3:
            logger.warning(f"Low confidence detection: {input_type} ({confidence:.2f})")
            
        # Get current context
        context = await self._get_current_context()
        
        # Calculate priority
        priority = self.priority_calculator.calculate(input_type, raw_input, context)
        
        # Create unified input
        unified_input = UnifiedInput(
            type=input_type,
            data=raw_input,
            metadata=input_metadata,
            priority=priority,
            context=context
        )
        
        # Route based on priority
        if priority in [Priority.CRITICAL, Priority.URGENT]:
            return await self._process_immediate(unified_input)
        else:
            return await self._queue_for_processing(unified_input)
    
    async def _process_immediate(self, input: UnifiedInput) -> Any:
        """Process critical/urgent inputs immediately"""
        logger.info(f"Immediate processing: {input.type} (Priority: {input.priority})")
        
        start_time = datetime.now()
        
        try:
            processor = self.processors.get(input.type)
            if not processor:
                logger.error(f"No processor for type: {input.type}")
                return {"error": "No processor available"}
                
            result = await processor.process(input.data, input.context)
            
            # Update metrics
            latency = (datetime.now() - start_time).total_seconds()
            self._update_metrics(input, latency)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {input.type}: {e}")
            return {"error": str(e)}
    
    async def _queue_for_processing(self, input: UnifiedInput) -> Dict[str, str]:
        """Queue input for background processing"""
        if len(self.priority_queue) >= self.max_queue_size:
            # Queue full - buffer lowest priority or drop
            if input.priority == Priority.LOW or input.priority == Priority.DEFERRED:
                self.buffer.append(input)
                return {"status": "buffered", "id": input.metadata.input_id}
            else:
                # Remove lowest priority item
                self._remove_lowest_priority()
                
        heapq.heappush(self.priority_queue, input)
        self.metrics['queue_size'] = len(self.priority_queue)
        
        return {"status": "queued", "id": input.metadata.input_id}
    
    async def _process_background(self):
        """Background processing of queued inputs"""
        while True:
            try:
                if self.priority_queue:
                    # Get highest priority input
                    input = heapq.heappop(self.priority_queue)
                    self.metrics['queue_size'] = len(self.priority_queue)
                    
                    # Check if still valid (not expired)
                    age = (datetime.now() - input.metadata.timestamp).total_seconds()
                    if age > 60:  # 1 minute expiry
                        logger.warning(f"Dropping expired input: {input.metadata.input_id}")
                        continue
                        
                    # Process
                    await self._process_immediate(input)
                    
                # Check buffer for items to promote
                if len(self.priority_queue) < self.max_queue_size / 2 and self.buffer:
                    buffered = self.buffer.popleft()
                    heapq.heappush(self.priority_queue, buffered)
                    
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _get_current_context(self) -> Dict[str, Any]:
        """Get current system context"""
        # This would integrate with your context management
        return {
            'user_state': {
                'flow_state': False,  # Would check actual state
                'stressed': False,
                'focused': True
            },
            'current_activity': 'working',
            'time_of_day': datetime.now().hour,
            'recent_inputs': list(self.metrics['by_type'].items())
        }
    
    def _generate_id(self) -> str:
        """Generate unique input ID"""
        return str(uuid.uuid4())
    
    def _update_metrics(self, input: UnifiedInput, latency: float):
        """Update processing metrics"""
        self.metrics['total_processed'] += 1
        self.metrics['by_type'][input.type] += 1
        self.metrics['by_priority'][input.priority] += 1
        
        # Update average latency
        n = self.metrics['total_processed']
        self.metrics['avg_latency'] = (
            (self.metrics['avg_latency'] * (n - 1) + latency) / n
        )
    
    def _remove_lowest_priority(self):
        """Remove lowest priority item from queue"""
        if not self.priority_queue:
            return
            
        # Find and remove lowest priority
        lowest_idx = 0
        lowest_priority = self.priority_queue[0].priority
        
        for i, input in enumerate(self.priority_queue):
            if input.priority.value > lowest_priority.value:
                lowest_idx = i
                lowest_priority = input.priority
                
        removed = self.priority_queue.pop(lowest_idx)
        heapq.heapify(self.priority_queue)
        
        # Add to buffer if there's space
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(removed)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        return self.metrics.copy()

# ============================================
# DEFAULT PROCESSOR
# ============================================

class DefaultProcessor:
    """Default processor for when specific processors aren't available"""
    
    def __init__(self, name: str):
        self.name = name
        
    async def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process input data"""
        return {
            "processor": self.name,
            "status": "processed",
            "data_keys": list(data.keys()) if isinstance(data, dict) else str(type(data)),
            "context": context.get('current_activity', 'unknown')
        }

# Export main components
__all__ = [
    'UnifiedInputPipeline',
    'InputType',
    'Priority',
    'UnifiedInput',
    'InputDetector',
    'PriorityCalculator'
]