#!/usr/bin/env python3
"""
JARVIS Phase 2: Vision Processing System
Processes visual inputs, screen context, and visual patterns
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import cv2
from PIL import Image, ImageGrab
import pytesseract
import json
import logging
from pathlib import Path
import hashlib
import io
import base64
from enum import Enum
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualElementType(Enum):
    """Types of visual elements"""
    TEXT = "text"
    BUTTON = "button"
    WINDOW = "window"
    ICON = "icon"
    IMAGE = "image"
    CHART = "chart"
    VIDEO = "video"
    FACE = "face"
    OBJECT = "object"

@dataclass
class VisualElement:
    """Detected visual element with properties"""
    element_id: str
    element_type: VisualElementType
    location: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    content: Any  # Text content, image data, etc.
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    screen_region: Optional[str] = None

@dataclass
class ScreenContext:
    """Current screen context and state"""
    active_window: str
    visible_elements: List[VisualElement]
    screen_hash: str
    timestamp: datetime
    focus_area: Optional[Tuple[int, int, int, int]] = None
    user_activity: Optional[str] = None

@dataclass
class VisualPattern:
    """Detected visual pattern or workflow"""
    pattern_id: str
    pattern_type: str  # workflow, layout, interaction
    elements: List[VisualElement]
    frequency: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class VisionProcessingSystem:
    """Advanced vision processing for screen analysis and visual intelligence"""
    
    def __init__(self, cache_dir: str = "./jarvis_vision_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Screen capture and analysis
        self.screen_history = deque(maxlen=100)
        self.element_cache = {}
        self.pattern_library = defaultdict(list)
        
        # Visual analysis models
        self.text_regions = []
        self.ui_elements = []
        self.color_palette = []
        
        # Activity tracking
        self.visual_workflows = defaultdict(list)
        self.attention_map = np.zeros((1080, 1920))  # Default Full HD
        self.gaze_patterns = []
        
        # Processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.processing_queue = asyncio.Queue()
        
        # Performance optimization
        self.frame_skip = 5  # Process every 5th frame for efficiency
        self.roi_enabled = True  # Region of Interest processing
        
        # Initialize OCR
        self._init_ocr()
        
    def _init_ocr(self):
        """Initialize OCR engine"""
        try:
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self.ocr_available = True
        except:
            logger.warning("Tesseract OCR not available. Text detection limited.")
            self.ocr_available = False
    
    async def capture_screen_context(self, region: Optional[Tuple[int, int, int, int]] = None) -> ScreenContext:
        """Capture and analyze current screen context"""
        # Capture screenshot
        screenshot = await self._capture_screenshot(region)
        
        # Get active window info
        active_window = await self._get_active_window()
        
        # Detect visual elements
        elements = await self._detect_visual_elements(screenshot)
        
        # Calculate screen hash for change detection
        screen_hash = self._calculate_image_hash(screenshot)
        
        # Determine focus area
        focus_area = await self._detect_focus_area(screenshot, elements)
        
        # Infer user activity
        user_activity = await self._infer_activity(elements, active_window)
        
        context = ScreenContext(
            active_window=active_window,
            visible_elements=elements,
            screen_hash=screen_hash,
            timestamp=datetime.now(),
            focus_area=focus_area,
            user_activity=user_activity
        )
        
        # Store in history
        self.screen_history.append(context)
        
        # Detect patterns
        await self._detect_visual_patterns(context)
        
        return context
    
    async def track_visual_workflow(self, context: ScreenContext) -> Optional[str]:
        """Track visual workflows and interactions"""
        if not self.screen_history:
            return None
        
        # Get recent contexts
        recent_contexts = list(self.screen_history)[-10:]
        
        # Extract workflow steps
        workflow_steps = []
        for ctx in recent_contexts:
            step = {
                'window': ctx.active_window,
                'activity': ctx.user_activity,
                'elements': len(ctx.visible_elements),
                'timestamp': ctx.timestamp
            }
            workflow_steps.append(step)
        
        # Identify workflow pattern
        workflow_id = self._identify_workflow_pattern(workflow_steps)
        
        if workflow_id:
            self.visual_workflows[workflow_id].append(workflow_steps)
            return workflow_id
        
        return None
    
    async def find_visual_element(self, element_type: VisualElementType,
                                content: Optional[str] = None,
                                confidence_threshold: float = 0.7) -> List[VisualElement]:
        """Find specific visual elements on screen"""
        # Capture current screen
        screenshot = await self._capture_screenshot()
        
        # Detect all elements
        all_elements = await self._detect_visual_elements(screenshot)
        
        # Filter by type and content
        matching_elements = []
        for element in all_elements:
            if element.element_type != element_type:
                continue
            
            if element.confidence < confidence_threshold:
                continue
            
            if content and isinstance(element.content, str):
                if content.lower() not in element.content.lower():
                    continue
            
            matching_elements.append(element)
        
        return matching_elements
    
    async def analyze_visual_changes(self, time_window: timedelta = timedelta(seconds=5)) -> Dict[str, Any]:
        """Analyze visual changes over time window"""
        if len(self.screen_history) < 2:
            return {'changes_detected': False}
        
        cutoff_time = datetime.now() - time_window
        recent_contexts = [ctx for ctx in self.screen_history 
                          if ctx.timestamp > cutoff_time]
        
        if len(recent_contexts) < 2:
            return {'changes_detected': False}
        
        changes = {
            'changes_detected': True,
            'window_changes': [],
            'element_changes': [],
            'activity_changes': [],
            'change_rate': 0.0
        }
        
        # Analyze window changes
        windows = [ctx.active_window for ctx in recent_contexts]
        window_changes = len(set(windows)) - 1
        changes['window_changes'] = window_changes
        
        # Analyze element changes
        for i in range(1, len(recent_contexts)):
            prev_elements = set(e.element_id for e in recent_contexts[i-1].visible_elements)
            curr_elements = set(e.element_id for e in recent_contexts[i].visible_elements)
            
            added = curr_elements - prev_elements
            removed = prev_elements - curr_elements
            
            if added or removed:
                changes['element_changes'].append({
                    'timestamp': recent_contexts[i].timestamp,
                    'added': len(added),
                    'removed': len(removed)
                })
        
        # Calculate change rate
        total_time = (recent_contexts[-1].timestamp - recent_contexts[0].timestamp).total_seconds()
        if total_time > 0:
            changes['change_rate'] = len(changes['element_changes']) / total_time
        
        return changes
    
    async def generate_attention_heatmap(self) -> np.ndarray:
        """Generate attention heatmap based on visual focus"""
        # Decay existing attention map
        self.attention_map *= 0.95
        
        # Add attention from recent contexts
        for context in list(self.screen_history)[-20:]:
            if context.focus_area:
                x, y, w, h = context.focus_area
                # Add Gaussian attention
                self._add_gaussian_attention(x + w//2, y + h//2, w, h)
            
            # Add attention for interactive elements
            for element in context.visible_elements:
                if element.element_type in [VisualElementType.BUTTON, 
                                          VisualElementType.TEXT]:
                    x, y, w, h = element.location
                    self._add_gaussian_attention(x + w//2, y + h//2, w//2, h//2)
        
        # Normalize
        if self.attention_map.max() > 0:
            self.attention_map /= self.attention_map.max()
        
        return self.attention_map
    
    async def extract_screen_text(self, region: Optional[Tuple[int, int, int, int]] = None) -> List[Dict[str, Any]]:
        """Extract all text from screen region"""
        screenshot = await self._capture_screenshot(region)
        
        if not self.ocr_available:
            return []
        
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        
        # Apply text detection
        text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        # Extract text regions
        text_regions = []
        n_boxes = len(text_data['text'])
        
        for i in range(n_boxes):
            if int(text_data['conf'][i]) > 30:  # Confidence threshold
                text = text_data['text'][i].strip()
                if text:
                    text_regions.append({
                        'text': text,
                        'location': (
                            text_data['left'][i],
                            text_data['top'][i],
                            text_data['width'][i],
                            text_data['height'][i]
                        ),
                        'confidence': text_data['conf'][i] / 100.0
                    })
        
        return text_regions
    
    async def detect_ui_patterns(self) -> List[VisualPattern]:
        """Detect common UI patterns and layouts"""
        if len(self.screen_history) < 5:
            return []
        
        patterns = []
        
        # Analyze recent screens for patterns
        recent_contexts = list(self.screen_history)[-20:]
        
        # Group by window
        window_groups = defaultdict(list)
        for ctx in recent_contexts:
            window_groups[ctx.active_window].append(ctx)
        
        # Detect patterns per window
        for window, contexts in window_groups.items():
            if len(contexts) >= 3:
                # Extract common element positions
                element_positions = defaultdict(list)
                
                for ctx in contexts:
                    for element in ctx.visible_elements:
                        key = (element.element_type, element.content[:20] if isinstance(element.content, str) else '')
                        element_positions[key].append(element.location)
                
                # Find stable UI elements
                for key, positions in element_positions.items():
                    if len(positions) >= len(contexts) * 0.7:  # Present in 70% of contexts
                        pattern = VisualPattern(
                            pattern_id=f"ui_{window}_{key[0].value}",
                            pattern_type='ui_layout',
                            elements=[],  # Would be populated with actual elements
                            frequency=len(positions),
                            confidence=len(positions) / len(contexts),
                            metadata={
                                'window': window,
                                'element_type': key[0].value,
                                'stable_position': self._calculate_average_position(positions)
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _capture_screenshot(self, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """Capture screenshot of screen or region"""
        try:
            if region:
                screenshot = ImageGrab.grab(bbox=region)
            else:
                screenshot = ImageGrab.grab()
            
            return screenshot
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            # Return blank image as fallback
            return Image.new('RGB', (1920, 1080), color='black')
    
    async def _get_active_window(self) -> str:
        """Get active window title"""
        # This would use platform-specific APIs
        # For now, return placeholder
        return "Application Window"
    
    async def _detect_visual_elements(self, screenshot: Image.Image) -> List[VisualElement]:
        """Detect visual elements in screenshot"""
        elements = []
        
        # Convert to numpy array
        img_array = np.array(screenshot)
        
        # Detect text regions if OCR available
        if self.ocr_available:
            text_regions = await self.extract_screen_text()
            for region in text_regions:
                element = VisualElement(
                    element_id=f"text_{hashlib.md5(region['text'].encode()).hexdigest()[:8]}",
                    element_type=VisualElementType.TEXT,
                    location=region['location'],
                    confidence=region['confidence'],
                    content=region['text'],
                    timestamp=datetime.now()
                )
                elements.append(element)
        
        # Detect UI elements using edge detection
        edges = cv2.Canny(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and classify contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small contours
            if w < 20 or h < 20:
                continue
            
            # Classify based on aspect ratio and size
            aspect_ratio = w / h
            element_type = self._classify_ui_element(aspect_ratio, w, h)
            
            element = VisualElement(
                element_id=f"{element_type.value}_{x}_{y}",
                element_type=element_type,
                location=(x, y, w, h),
                confidence=0.7,  # Default confidence
                content=None,
                timestamp=datetime.now()
            )
            elements.append(element)
        
        # Limit number of elements for performance
        elements = elements[:50]
        
        return elements
    
    def _classify_ui_element(self, aspect_ratio: float, width: int, height: int) -> VisualElementType:
        """Classify UI element based on shape"""
        if 0.8 < aspect_ratio < 1.2 and width < 100:
            return VisualElementType.ICON
        elif aspect_ratio > 2 and height < 50:
            return VisualElementType.BUTTON
        elif width > 200 and height > 200:
            return VisualElementType.WINDOW
        else:
            return VisualElementType.IMAGE
    
    async def _detect_focus_area(self, screenshot: Image.Image, 
                               elements: List[VisualElement]) -> Optional[Tuple[int, int, int, int]]:
        """Detect area of visual focus"""
        if not elements:
            return None
        
        # Find area with most elements
        if len(elements) > 5:
            # Use k-means clustering on element centers
            centers = np.array([(e.location[0] + e.location[2]//2, 
                               e.location[1] + e.location[3]//2) 
                              for e in elements])
            
            kmeans = KMeans(n_clusters=min(3, len(elements)//3))
            labels = kmeans.fit_predict(centers)
            
            # Find largest cluster
            unique, counts = np.unique(labels, return_counts=True)
            largest_cluster = unique[np.argmax(counts)]
            
            # Get bounding box of largest cluster
            cluster_elements = [e for i, e in enumerate(elements) 
                              if labels[i] == largest_cluster]
            
            if cluster_elements:
                x_coords = [e.location[0] for e in cluster_elements]
                y_coords = [e.location[1] for e in cluster_elements]
                x2_coords = [e.location[0] + e.location[2] for e in cluster_elements]
                y2_coords = [e.location[1] + e.location[3] for e in cluster_elements]
                
                return (min(x_coords), min(y_coords), 
                       max(x2_coords) - min(x_coords), 
                       max(y2_coords) - min(y_coords))
        
        return None
    
    async def _infer_activity(self, elements: List[VisualElement], 
                            window: str) -> Optional[str]:
        """Infer user activity from visual elements"""
        # Simple heuristic-based inference
        text_content = ' '.join([e.content for e in elements 
                               if e.element_type == VisualElementType.TEXT 
                               and isinstance(e.content, str)])
        
        # Check for common activities
        activities = {
            'coding': ['def', 'class', 'import', 'function', '{', '}'],
            'browsing': ['http', 'www', 'search', 'google'],
            'email': ['inbox', 'compose', 'reply', 'subject'],
            'document': ['page', 'paragraph', 'chapter', 'section'],
            'chat': ['message', 'send', 'typing', 'online']
        }
        
        for activity, keywords in activities.items():
            if any(kw in text_content.lower() for kw in keywords):
                return activity
        
        return 'general'
    
    def _calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate perceptual hash of image"""
        # Resize to 8x8
        small = image.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
        
        # Calculate mean
        pixels = list(small.getdata())
        avg = sum(pixels) / len(pixels)
        
        # Create hash
        bits = ''.join(['1' if pixel > avg else '0' for pixel in pixels])
        return hex(int(bits, 2))[2:].zfill(16)
    
    async def _detect_visual_patterns(self, context: ScreenContext):
        """Detect patterns in visual interactions"""
        # Store element sequences
        if len(self.screen_history) >= 2:
            prev_context = self.screen_history[-2]
            
            # Check for repeated element interactions
            prev_elements = {e.element_id for e in prev_context.visible_elements}
            curr_elements = {e.element_id for e in context.visible_elements}
            
            # Find persistent elements
            persistent = prev_elements & curr_elements
            
            if len(persistent) > 5:  # Significant overlap
                pattern = VisualPattern(
                    pattern_id=f"interact_{context.active_window}_{len(self.pattern_library)}",
                    pattern_type='interaction',
                    elements=[e for e in context.visible_elements 
                             if e.element_id in persistent],
                    frequency=1,
                    confidence=len(persistent) / len(curr_elements),
                    metadata={
                        'window': context.active_window,
                        'timestamp': context.timestamp
                    }
                )
                
                # Check if similar pattern exists
                similar_found = False
                for existing_patterns in self.pattern_library.values():
                    for existing in existing_patterns:
                        if self._patterns_similar(pattern, existing):
                            existing.frequency += 1
                            existing.confidence = min(existing.confidence * 1.1, 1.0)
                            similar_found = True
                            break
                
                if not similar_found:
                    self.pattern_library[pattern.pattern_type].append(pattern)
    
    def _patterns_similar(self, p1: VisualPattern, p2: VisualPattern) -> bool:
        """Check if two patterns are similar"""
        if p1.pattern_type != p2.pattern_type:
            return False
        
        if p1.metadata.get('window') != p2.metadata.get('window'):
            return False
        
        # Check element overlap
        p1_types = Counter(e.element_type for e in p1.elements)
        p2_types = Counter(e.element_type for e in p2.elements)
        
        # Calculate similarity
        common_types = sum((p1_types & p2_types).values())
        total_types = sum((p1_types | p2_types).values())
        
        similarity = common_types / total_types if total_types > 0 else 0
        
        return similarity > 0.7
    
    def _identify_workflow_pattern(self, steps: List[Dict[str, Any]]) -> Optional[str]:
        """Identify workflow pattern from steps"""
        if len(steps) < 3:
            return None
        
        # Create workflow signature
        signature = []
        for step in steps:
            sig = f"{step['window']}_{step['activity']}"
            signature.append(sig)
        
        # Check against known workflows
        workflow_id = '_'.join(signature[:3])  # Use first 3 steps as ID
        
        return workflow_id
    
    def _add_gaussian_attention(self, cx: int, cy: int, w: int, h: int):
        """Add Gaussian attention to attention map"""
        y, x = np.ogrid[:self.attention_map.shape[0], :self.attention_map.shape[1]]
        
        # Create Gaussian
        gaussian = np.exp(-((x - cx)**2 / (2 * w**2) + (y - cy)**2 / (2 * h**2)))
        
        # Add to attention map
        self.attention_map += gaussian * 0.1
    
    def _calculate_average_position(self, positions: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Calculate average position from list of positions"""
        if not positions:
            return (0, 0, 0, 0)
        
        avg_x = sum(p[0] for p in positions) // len(positions)
        avg_y = sum(p[1] for p in positions) // len(positions)
        avg_w = sum(p[2] for p in positions) // len(positions)
        avg_h = sum(p[3] for p in positions) // len(positions)
        
        return (avg_x, avg_y, avg_w, avg_h)
    
    async def save_visual_snapshot(self, context: ScreenContext, 
                                 description: str = "") -> str:
        """Save visual snapshot for later reference"""
        snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        snapshot_path = self.cache_dir / f"{snapshot_id}.json"
        
        # Prepare snapshot data
        snapshot_data = {
            'id': snapshot_id,
            'timestamp': context.timestamp.isoformat(),
            'description': description,
            'active_window': context.active_window,
            'screen_hash': context.screen_hash,
            'focus_area': context.focus_area,
            'user_activity': context.user_activity,
            'element_count': len(context.visible_elements),
            'elements': [
                {
                    'type': e.element_type.value,
                    'location': e.location,
                    'content': e.content[:50] if isinstance(e.content, str) else None,
                    'confidence': e.confidence
                }
                for e in context.visible_elements[:20]  # Limit to 20 elements
            ]
        }
        
        # Save to file
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        logger.info(f"Saved visual snapshot: {snapshot_id}")
        return snapshot_id
    
    async def load_visual_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Load previously saved visual snapshot"""
        snapshot_path = self.cache_dir / f"{snapshot_id}.json"
        
        if not snapshot_path.exists():
            return None
        
        with open(snapshot_path, 'r') as f:
            return json.load(f)

# Create singleton instance
_vision_system = None

async def get_vision_system() -> VisionProcessingSystem:
    """Get or create vision processing system"""
    global _vision_system
    if _vision_system is None:
        _vision_system = VisionProcessingSystem()
    return _vision_system
