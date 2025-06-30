# JARVIS Phase 2 Integration Guide

## Overview

Phase 2 builds upon Phase 1's foundation to add intelligent processing capabilities:
- **Context Persistence**: Maintains conversation and activity context between interactions
- **Predictive Pre-loading**: Pre-loads likely next actions based on patterns
- **Temporal Processing**: Advanced time-series analysis and pattern recognition
- **Vision Processing**: Screen analysis and visual intelligence

## Quick Start

```python
from phase2.jarvis_phase2_core import create_jarvis_phase2, Phase2Config

# Configure Phase 2 features
config = Phase2Config(
    enable_context_persistence=True,
    enable_predictive_preload=True,
    enable_temporal_processing=True,
    enable_vision_processing=True
)

# Initialize JARVIS Phase 2
jarvis = await create_jarvis_phase2(config)

# Process with full intelligence
result = await jarvis.process_with_intelligence({
    'query': 'Schedule a meeting for tomorrow',
    'source': 'voice'
})
```

## Component Integration

### 1. Context Persistence

The context system maintains hierarchical memory:

```python
from phase2.context_persistence import get_context_persistence

context_system = await get_context_persistence()

# Add context
await context_system.add_context(
    context_type='conversation',
    content='User prefers morning meetings',
    source='observed'
)

# Retrieve relevant context
relevant = await context_system.get_relevant_context(
    query='When should we schedule?',
    time_window=timedelta(days=7)
)
```

**Memory Hierarchy:**
- **Working Memory**: Last 10 items (immediate context)
- **Short-term Memory**: Last hour of context
- **Long-term Memory**: Persistent patterns and preferences

### 2. Predictive Pre-loading

The predictive system learns from user behavior:

```python
from phase2.predictive_preloading import get_predictive_system

predictive_system = await get_predictive_system()

# Record user actions
await predictive_system.record_action(
    action_type='app_launch',
    target='Chrome',
    context={'time_of_day': 'morning'}
)

# Get predictions
predictions = await predictive_system.predict_next_actions(
    current_context={'time_of_day': 'morning'},
    top_k=5
)

# Pre-load resources
preloaded = await predictive_system.pre_load_resources(predictions)
```

**Prediction Methods:**
- ML-based predictions (Random Forest)
- Pattern matching
- Temporal patterns
- Sequential patterns

### 3. Temporal Processing

Advanced time-series analysis:

```python
from phase2.temporal_processing import get_temporal_system

temporal_system = await get_temporal_system()

# Add temporal events
await temporal_system.add_temporal_event(
    event_type='work_start',
    value={'location': 'office'},
    duration=timedelta(hours=8)
)

# Detect routines
routines = await temporal_system.detect_routines(lookback_days=30)

# Find correlations
correlations = await temporal_system.find_temporal_correlations(
    event_types=['coffee', 'productivity'],
    lag_range=(-10, 10)
)
```

**Pattern Detection:**
- Periodic patterns (FFT-based)
- Trending patterns
- Clustering patterns
- Daily/weekly routines

### 4. Vision Processing

Screen and visual analysis:

```python
from phase2.vision_processing import get_vision_system

vision_system = await get_vision_system()

# Capture screen context
context = await vision_system.capture_screen_context()

# Find specific elements
buttons = await vision_system.find_visual_element(
    element_type=VisualElementType.BUTTON,
    content='Submit'
)

# Generate attention heatmap
heatmap = await vision_system.generate_attention_heatmap()
```

**Visual Features:**
- OCR text extraction
- UI element detection
- Visual workflow tracking
- Attention mapping

## Integration with Existing JARVIS

### Enhancing Phase 1 Components

Phase 2 enhances Phase 1 components automatically:

```python
# Phase 1 pipeline now includes context
result = await jarvis.enhanced_core.process_input(
    input_data,
    source='user'
)
# Context is automatically included in processing
```

### Adding to Existing Workflows

```python
# Your existing JARVIS code
from your_jarvis import YourJARVIS

# Add Phase 2 intelligence
from phase2.jarvis_phase2_core import JARVISPhase2Core

class EnhancedJARVIS(YourJARVIS):
    def __init__(self):
        super().__init__()
        self.phase2 = JARVISPhase2Core()
    
    async def process_request(self, request):
        # Gather intelligent context
        context = await self.phase2._gather_intelligent_context(request)
        
        # Your existing processing with context
        result = await super().process_request({
            **request,
            'context': context
        })
        
        # Learn from interaction
        await self.phase2.context_system.add_context(
            'interaction',
            result
        )
        
        return result
```

## Configuration Options

### Context Persistence
```python
config.context_memory_size = 1000  # Max items in memory
config.context_relevance_threshold = 0.6  # Min relevance score
```

### Predictive Pre-loading
```python
config.prediction_confidence_threshold = 0.5  # Min confidence
config.max_preload_items = 5  # Max items to pre-load
```

### Temporal Processing
```python
config.temporal_pattern_min_confidence = 0.7
config.temporal_anomaly_threshold = 3.0  # Std deviations
```

### Vision Processing
```python
config.vision_capture_interval = 2.0  # Seconds between captures
config.vision_roi_enabled = True  # Region of Interest processing
```

## Performance Considerations

### Resource Usage
- **Memory**: ~500MB for full context history
- **CPU**: Background processing uses 5-10%
- **Storage**: SQLite DB grows ~1MB/day

### Optimization Tips

1. **Disable unused components**:
```python
config = Phase2Config(
    enable_vision_processing=False  # If not needed
)
```

2. **Adjust memory limits**:
```python
context_system.memory.working = deque(maxlen=5)  # Smaller working memory
```

3. **Reduce prediction frequency**:
```python
predictive_system.frame_skip = 10  # Process every 10th event
```

## Troubleshooting

### Common Issues

1. **OCR not working**:
   - Install Tesseract: `brew install tesseract`
   - Set path: `export TESSDATA_PREFIX=/usr/local/share/tessdata`

2. **Vision capture fails**:
   - Check display permissions
   - Try with reduced region: `capture_screen_context(region=(0,0,800,600))`

3. **High memory usage**:
   - Clear old context: `context_system.memory.long_term.clear()`
   - Reduce cache size in config

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Example 1: Smart Assistant
```python
# Remember user preferences
await context_system.add_context(
    'preference',
    'likes_dark_theme',
    metadata={'ui_preference': True}
)

# Later, when changing UI
prefs = await context_system.get_user_preferences()
if 'likes_dark_theme' in prefs.get('ui', {}):
    apply_dark_theme()
```

### Example 2: Workflow Automation
```python
# Track workflow
workflow = await vision_system.track_visual_workflow(screen_context)

# Predict next step
predictions = await predictive_system.predict_next_actions({
    'workflow': workflow,
    'current_app': screen_context.active_window
})

# Pre-open predicted app
if predictions[0].action_type == 'app_launch':
    await launch_app(predictions[0].target)
```

### Example 3: Anomaly Detection
```python
# Monitor system events
await temporal_system.add_temporal_event(
    'cpu_usage',
    value=current_cpu_percent
)

# Detect anomalies
if await temporal_system._is_anomaly(event):
    alert_user("Unusual CPU usage detected")
```

## Next Steps

1. **Run the demo**: `python phase2/launch_phase2_demo.py`
2. **Run tests**: `python phase2/test_phase2.py`
3. **Integrate with your code**: Follow examples above
4. **Proceed to Phase 3**: Natural language and emotional intelligence

## Support

For issues or questions:
- Check logs in `./jarvis_logs/`
- Review test cases in `test_phase2.py`
- Refer to component docstrings
