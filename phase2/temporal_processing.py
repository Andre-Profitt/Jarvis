#!/usr/bin/env python3
"""
JARVIS Phase 2: Temporal Processing System
Advanced time-series analysis and temporal pattern recognition
"""

import asyncio
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import json
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalGranularity(Enum):
    """Time granularity levels"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    SEASON = "season"

@dataclass
class TemporalEvent:
    """Single temporal event with metadata"""
    event_id: str
    event_type: str
    timestamp: datetime
    value: Any
    duration: Optional[timedelta] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalPattern:
    """Detected temporal pattern"""
    pattern_id: str
    pattern_type: str  # periodic, trending, anomaly, cluster
    confidence: float
    period: Optional[timedelta] = None
    trend: Optional[str] = None  # increasing, decreasing, stable
    events: List[TemporalEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TimeWindow:
    """Flexible time window for analysis"""
    start: datetime
    end: datetime
    granularity: TemporalGranularity
    
    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp <= self.end
    
    def duration(self) -> timedelta:
        return self.end - self.start

class TemporalProcessingSystem:
    """Advanced temporal processing and pattern recognition"""
    
    def __init__(self):
        # Event storage
        self.events = defaultdict(list)  # event_type -> List[TemporalEvent]
        self.event_streams = defaultdict(deque)  # Real-time streams
        
        # Pattern detection
        self.detected_patterns = []
        self.pattern_templates = self._load_pattern_templates()
        
        # Time series analysis
        self.time_series_cache = {}
        self.anomaly_detectors = {}
        
        # Temporal predictions
        self.prediction_models = {}
        
        # Activity tracking
        self.activity_windows = defaultdict(list)
        self.routine_patterns = {}
        
        # Configuration
        self.min_pattern_confidence = 0.7
        self.anomaly_threshold = 3.0  # Standard deviations
        
    async def add_temporal_event(self, event_type: str, value: Any,
                               timestamp: Optional[datetime] = None,
                               duration: Optional[timedelta] = None,
                               metadata: Optional[Dict] = None) -> str:
        """Add a temporal event for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        event = TemporalEvent(
            event_id=f"{event_type}_{timestamp.timestamp()}",
            event_type=event_type,
            timestamp=timestamp,
            value=value,
            duration=duration,
            metadata=metadata or {}
        )
        
        # Store event
        self.events[event_type].append(event)
        self.event_streams[event_type].append(event)
        
        # Limit stream size
        if len(self.event_streams[event_type]) > 1000:
            self.event_streams[event_type].popleft()
        
        # Trigger pattern detection if enough events
        if len(self.events[event_type]) % 10 == 0:
            await self._detect_patterns(event_type)
        
        # Check for anomalies
        if await self._is_anomaly(event):
            await self._handle_anomaly(event)
        
        return event.event_id
    
    async def get_temporal_context(self, time_window: Optional[TimeWindow] = None,
                                 event_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get comprehensive temporal context"""
        if time_window is None:
            # Default to last 24 hours
            time_window = TimeWindow(
                start=datetime.now() - timedelta(days=1),
                end=datetime.now(),
                granularity=TemporalGranularity.HOUR
            )
        
        context = {
            'time_window': {
                'start': time_window.start.isoformat(),
                'end': time_window.end.isoformat(),
                'duration': str(time_window.duration())
            },
            'events': {},
            'patterns': [],
            'statistics': {},
            'predictions': {}
        }
        
        # Gather events in window
        for event_type, events in self.events.items():
            if event_types and event_type not in event_types:
                continue
            
            window_events = [e for e in events if time_window.contains(e.timestamp)]
            if window_events:
                context['events'][event_type] = self._summarize_events(window_events)
        
        # Find relevant patterns
        context['patterns'] = [
            self._pattern_to_dict(p) for p in self.detected_patterns
            if any(time_window.contains(e.timestamp) for e in p.events)
        ]
        
        # Calculate statistics
        context['statistics'] = await self._calculate_temporal_statistics(
            time_window, event_types
        )
        
        # Generate predictions
        context['predictions'] = await self._generate_temporal_predictions(
            time_window, event_types
        )
        
        return context
    
    async def detect_routines(self, lookback_days: int = 30) -> List[Dict[str, Any]]:
        """Detect user routines and habits"""
        routines = []
        
        # Analyze daily patterns
        daily_patterns = await self._analyze_daily_patterns(lookback_days)
        routines.extend(daily_patterns)
        
        # Analyze weekly patterns
        weekly_patterns = await self._analyze_weekly_patterns(lookback_days)
        routines.extend(weekly_patterns)
        
        # Analyze activity sequences
        sequence_patterns = await self._analyze_sequence_patterns(lookback_days)
        routines.extend(sequence_patterns)
        
        # Store detected routines
        for routine in routines:
            self.routine_patterns[routine['id']] = routine
        
        return routines
    
    async def predict_next_temporal_event(self, event_type: str,
                                        horizon: timedelta = timedelta(hours=1)) -> List[Dict[str, Any]]:
        """Predict next occurrences of temporal events"""
        predictions = []
        
        if event_type not in self.events:
            return predictions
        
        events = self.events[event_type]
        if len(events) < 5:
            return predictions
        
        # Method 1: Periodic pattern prediction
        periodic_pred = await self._predict_periodic(events, horizon)
        predictions.extend(periodic_pred)
        
        # Method 2: Trend-based prediction
        trend_pred = await self._predict_trend(events, horizon)
        predictions.extend(trend_pred)
        
        # Method 3: ML-based prediction (if model exists)
        if event_type in self.prediction_models:
            ml_pred = await self._predict_ml(event_type, horizon)
            predictions.extend(ml_pred)
        
        # Sort by confidence
        predictions.sort(key=lambda p: p['confidence'], reverse=True)
        
        return predictions[:5]  # Top 5 predictions
    
    async def analyze_time_series(self, event_type: str,
                                metric: str = 'value',
                                window: Optional[TimeWindow] = None) -> Dict[str, Any]:
        """Perform time series analysis on event stream"""
        if event_type not in self.events:
            return {}
        
        events = self.events[event_type]
        if window:
            events = [e for e in events if window.contains(e.timestamp)]
        
        if len(events) < 2:
            return {}
        
        # Convert to pandas series
        timestamps = [e.timestamp for e in events]
        values = [self._extract_metric(e, metric) for e in events]
        
        ts = pd.Series(values, index=pd.DatetimeIndex(timestamps))
        ts = ts.sort_index()
        
        analysis = {
            'summary': {
                'count': len(ts),
                'mean': float(ts.mean()),
                'std': float(ts.std()),
                'min': float(ts.min()),
                'max': float(ts.max())
            }
        }
        
        # Trend analysis
        if len(ts) >= 10:
            analysis['trend'] = self._analyze_trend(ts)
        
        # Seasonality detection
        if len(ts) >= 24:  # At least 24 data points
            analysis['seasonality'] = self._detect_seasonality(ts)
        
        # Autocorrelation
        if len(ts) >= 20:
            analysis['autocorrelation'] = self._calculate_autocorrelation(ts)
        
        # Stationarity test
        analysis['is_stationary'] = self._test_stationarity(ts)
        
        return analysis
    
    async def find_temporal_correlations(self, event_types: List[str],
                                       lag_range: Tuple[int, int] = (-10, 10)) -> List[Dict[str, Any]]:
        """Find correlations between different temporal event streams"""
        correlations = []
        
        # Check each pair of event types
        for i, type1 in enumerate(event_types):
            for type2 in event_types[i+1:]:
                if type1 not in self.events or type2 not in self.events:
                    continue
                
                corr = await self._calculate_cross_correlation(
                    self.events[type1],
                    self.events[type2],
                    lag_range
                )
                
                if corr and abs(corr['max_correlation']) > 0.5:
                    correlations.append(corr)
        
        # Sort by correlation strength
        correlations.sort(key=lambda c: abs(c['max_correlation']), reverse=True)
        
        return correlations
    
    async def _detect_patterns(self, event_type: str):
        """Detect patterns in temporal events"""
        events = self.events[event_type]
        
        if len(events) < 10:
            return
        
        # Detect periodic patterns
        periodic_patterns = await self._detect_periodic_patterns(events)
        self.detected_patterns.extend(periodic_patterns)
        
        # Detect trending patterns
        trending_patterns = await self._detect_trending_patterns(events)
        self.detected_patterns.extend(trending_patterns)
        
        # Detect clustering patterns
        cluster_patterns = await self._detect_cluster_patterns(events)
        self.detected_patterns.extend(cluster_patterns)
        
        # Remove duplicate or overlapping patterns
        self._consolidate_patterns()
    
    async def _detect_periodic_patterns(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """Detect periodic/cyclical patterns"""
        patterns = []
        
        # Convert to time series
        timestamps = np.array([e.timestamp.timestamp() for e in events])
        values = np.array([self._extract_numeric_value(e.value) for e in events])
        
        # Check different period lengths
        periods_to_check = [
            timedelta(minutes=30),
            timedelta(hours=1),
            timedelta(hours=24),
            timedelta(days=7)
        ]
        
        for period in periods_to_check:
            period_seconds = period.total_seconds()
            
            # Use FFT to detect periodicity
            if len(values) >= 10:
                fft = np.fft.fft(values)
                frequencies = np.fft.fftfreq(len(values), d=np.mean(np.diff(timestamps)))
                
                # Find dominant frequency close to expected period
                expected_freq = 1.0 / period_seconds
                freq_tolerance = expected_freq * 0.2  # 20% tolerance
                
                mask = np.abs(frequencies - expected_freq) < freq_tolerance
                if np.any(mask):
                    power = np.abs(fft[mask])
                    if np.max(power) > np.mean(np.abs(fft)) * 2:  # Strong periodic signal
                        pattern = TemporalPattern(
                            pattern_id=f"periodic_{events[0].event_type}_{period}",
                            pattern_type='periodic',
                            confidence=min(np.max(power) / np.mean(np.abs(fft)), 1.0),
                            period=period,
                            events=events[-20:],  # Last 20 events
                            metadata={'fft_power': float(np.max(power))}
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_trending_patterns(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """Detect trending patterns (increasing/decreasing)"""
        patterns = []
        
        if len(events) < 5:
            return patterns
        
        # Analyze recent events
        recent_events = events[-20:]
        timestamps = np.array([e.timestamp.timestamp() for e in recent_events])
        values = np.array([self._extract_numeric_value(e.value) for e in recent_events])
        
        # Linear regression for trend
        z = np.polyfit(timestamps, values, 1)
        slope = z[0]
        
        # Determine trend strength
        residuals = values - np.polyval(z, timestamps)
        r_squared = 1 - (np.sum(residuals**2) / np.sum((values - np.mean(values))**2))
        
        if abs(r_squared) > 0.5:  # Decent fit
            trend_type = 'increasing' if slope > 0 else 'decreasing'
            
            pattern = TemporalPattern(
                pattern_id=f"trend_{events[0].event_type}_{trend_type}",
                pattern_type='trending',
                confidence=abs(r_squared),
                trend=trend_type,
                events=recent_events,
                metadata={
                    'slope': float(slope),
                    'r_squared': float(r_squared)
                }
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_cluster_patterns(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """Detect temporal clustering patterns"""
        patterns = []
        
        if len(events) < 10:
            return patterns
        
        # Convert timestamps to numeric features
        timestamps = np.array([e.timestamp.timestamp() for e in events])
        
        # Normalize timestamps
        scaler = StandardScaler()
        timestamps_scaled = scaler.fit_transform(timestamps.reshape(-1, 1))
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.3, min_samples=3)
        labels = clustering.fit_predict(timestamps_scaled)
        
        # Extract clusters
        unique_labels = set(labels) - {-1}  # Exclude noise
        
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_events = [events[i] for i in cluster_indices]
            
            if len(cluster_events) >= 3:
                pattern = TemporalPattern(
                    pattern_id=f"cluster_{events[0].event_type}_{label}",
                    pattern_type='cluster',
                    confidence=len(cluster_events) / len(events),
                    events=cluster_events,
                    metadata={
                        'cluster_size': len(cluster_events),
                        'time_span': (
                            max(e.timestamp for e in cluster_events) -
                            min(e.timestamp for e in cluster_events)
                        ).total_seconds()
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_daily_patterns(self, lookback_days: int) -> List[Dict[str, Any]]:
        """Analyze patterns within daily cycles"""
        patterns = []
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        # Group events by hour of day
        hourly_events = defaultdict(list)
        
        for event_type, events in self.events.items():
            for event in events:
                if event.timestamp > cutoff:
                    hour = event.timestamp.hour
                    hourly_events[hour].append({
                        'type': event_type,
                        'event': event
                    })
        
        # Find consistent hourly patterns
        for hour, hour_events in hourly_events.items():
            if len(hour_events) >= lookback_days * 0.5:  # At least 50% of days
                event_types = Counter(e['type'] for e in hour_events)
                
                for event_type, count in event_types.most_common(3):
                    if count >= lookback_days * 0.3:  # 30% consistency
                        patterns.append({
                            'id': f"daily_{hour}_{event_type}",
                            'type': 'daily_routine',
                            'hour': hour,
                            'event_type': event_type,
                            'frequency': count / lookback_days,
                            'confidence': min(count / lookback_days, 1.0),
                            'description': f"{event_type} typically occurs at {hour:02d}:00"
                        })
        
        return patterns
    
    async def _analyze_weekly_patterns(self, lookback_days: int) -> List[Dict[str, Any]]:
        """Analyze patterns within weekly cycles"""
        patterns = []
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        # Group events by day of week
        daily_events = defaultdict(list)
        
        for event_type, events in self.events.items():
            for event in events:
                if event.timestamp > cutoff:
                    dow = event.timestamp.weekday()
                    daily_events[dow].append({
                        'type': event_type,
                        'event': event
                    })
        
        # Find consistent weekly patterns
        for dow, day_events in daily_events.items():
            weeks = lookback_days / 7
            if len(day_events) >= weeks * 0.5:  # At least 50% of weeks
                event_types = Counter(e['type'] for e in day_events)
                
                for event_type, count in event_types.most_common(3):
                    if count >= weeks * 0.3:  # 30% consistency
                        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                   'Friday', 'Saturday', 'Sunday']
                        patterns.append({
                            'id': f"weekly_{dow}_{event_type}",
                            'type': 'weekly_routine',
                            'day_of_week': dow,
                            'day_name': day_names[dow],
                            'event_type': event_type,
                            'frequency': count / weeks,
                            'confidence': min(count / weeks, 1.0),
                            'description': f"{event_type} typically occurs on {day_names[dow]}"
                        })
        
        return patterns
    
    async def _analyze_sequence_patterns(self, lookback_days: int) -> List[Dict[str, Any]]:
        """Analyze sequential patterns in activities"""
        patterns = []
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        # Build sequences of events
        all_events = []
        for event_type, events in self.events.items():
            for event in events:
                if event.timestamp > cutoff:
                    all_events.append((event.timestamp, event_type, event))
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x[0])
        
        # Find common sequences
        sequences = defaultdict(int)
        for i in range(len(all_events) - 2):
            # Look for 3-event sequences within 1 hour
            if (all_events[i+2][0] - all_events[i][0]).total_seconds() < 3600:
                seq = tuple(e[1] for e in all_events[i:i+3])
                sequences[seq] += 1
        
        # Extract significant sequences
        for seq, count in sequences.items():
            if count >= 5:  # At least 5 occurrences
                patterns.append({
                    'id': f"sequence_{'_'.join(seq)}",
                    'type': 'activity_sequence',
                    'sequence': list(seq),
                    'count': count,
                    'confidence': min(count / 20, 1.0),
                    'description': f"Pattern: {' → '.join(seq)}"
                })
        
        return sorted(patterns, key=lambda p: p['confidence'], reverse=True)[:10]
    
    def _extract_numeric_value(self, value: Any) -> float:
        """Extract numeric value from various types"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, dict) and 'value' in value:
            return self._extract_numeric_value(value['value'])
        elif isinstance(value, str):
            try:
                return float(value)
            except:
                return hash(value) % 100  # Convert to pseudo-numeric
        else:
            return 0.0
    
    def _summarize_events(self, events: List[TemporalEvent]) -> Dict[str, Any]:
        """Summarize a list of temporal events"""
        return {
            'count': len(events),
            'first': events[0].timestamp.isoformat(),
            'last': events[-1].timestamp.isoformat(),
            'duration': (events[-1].timestamp - events[0].timestamp).total_seconds(),
            'values': [e.value for e in events[-5:]]  # Last 5 values
        }
    
    def _pattern_to_dict(self, pattern: TemporalPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary"""
        return {
            'id': pattern.pattern_id,
            'type': pattern.pattern_type,
            'confidence': pattern.confidence,
            'period': str(pattern.period) if pattern.period else None,
            'trend': pattern.trend,
            'event_count': len(pattern.events),
            'metadata': pattern.metadata
        }
    
    async def _is_anomaly(self, event: TemporalEvent) -> bool:
        """Check if event is anomalous"""
        if event.event_type not in self.events:
            return False
        
        historical = self.events[event.event_type]
        if len(historical) < 20:
            return False
        
        # Get recent values
        recent_values = [self._extract_numeric_value(e.value) 
                        for e in historical[-20:]]
        
        # Calculate statistics
        mean = np.mean(recent_values)
        std = np.std(recent_values)
        
        # Check if current value is anomalous
        current_value = self._extract_numeric_value(event.value)
        z_score = abs((current_value - mean) / (std + 1e-6))
        
        return z_score > self.anomaly_threshold
    
    async def _handle_anomaly(self, event: TemporalEvent):
        """Handle detected anomaly"""
        logger.warning(f"Anomaly detected in {event.event_type}: {event.value}")
        
        # Create anomaly pattern
        pattern = TemporalPattern(
            pattern_id=f"anomaly_{event.event_id}",
            pattern_type='anomaly',
            confidence=0.9,
            events=[event],
            metadata={
                'anomaly_type': 'statistical',
                'detected_at': datetime.now().isoformat()
            }
        )
        
        self.detected_patterns.append(pattern)
    
    def _load_pattern_templates(self) -> Dict[str, Any]:
        """Load predefined pattern templates"""
        return {
            'morning_routine': {
                'time_range': (time(6, 0), time(10, 0)),
                'expected_events': ['wake', 'coffee', 'email_check']
            },
            'work_hours': {
                'time_range': (time(9, 0), time(17, 0)),
                'expected_events': ['work_start', 'meetings', 'work_end']
            },
            'evening_routine': {
                'time_range': (time(18, 0), time(23, 0)),
                'expected_events': ['dinner', 'relax', 'sleep_prep']
            }
        }
    
    async def _calculate_temporal_statistics(self, window: TimeWindow,
                                           event_types: Optional[List[str]]) -> Dict[str, Any]:
        """Calculate statistics for time window"""
        stats = {}
        
        for event_type, events in self.events.items():
            if event_types and event_type not in event_types:
                continue
            
            window_events = [e for e in events if window.contains(e.timestamp)]
            
            if window_events:
                stats[event_type] = {
                    'count': len(window_events),
                    'rate_per_hour': len(window_events) / (window.duration().total_seconds() / 3600),
                    'average_interval': np.mean([
                        (window_events[i+1].timestamp - window_events[i].timestamp).total_seconds()
                        for i in range(len(window_events)-1)
                    ]) if len(window_events) > 1 else None
                }
        
        return stats
    
    async def _generate_temporal_predictions(self, window: TimeWindow,
                                           event_types: Optional[List[str]]) -> Dict[str, Any]:
        """Generate predictions for time window"""
        predictions = {}
        
        for event_type in (event_types or self.events.keys()):
            if event_type in self.events:
                next_events = await self.predict_next_temporal_event(
                    event_type, 
                    window.duration()
                )
                if next_events:
                    predictions[event_type] = next_events
        
        return predictions
    
    async def _predict_periodic(self, events: List[TemporalEvent],
                              horizon: timedelta) -> List[Dict[str, Any]]:
        """Predict based on periodic patterns"""
        predictions = []
        
        # Find periodic patterns for this event type
        periodic_patterns = [
            p for p in self.detected_patterns
            if p.pattern_type == 'periodic' and 
            any(e.event_type == events[0].event_type for e in p.events)
        ]
        
        for pattern in periodic_patterns:
            if pattern.period:
                # Calculate next occurrence
                last_event = max(pattern.events, key=lambda e: e.timestamp)
                next_time = last_event.timestamp + pattern.period
                
                while next_time < datetime.now() + horizon:
                    if next_time > datetime.now():
                        predictions.append({
                            'timestamp': next_time.isoformat(),
                            'confidence': pattern.confidence * 0.8,  # Decay confidence
                            'basis': 'periodic_pattern',
                            'period': str(pattern.period)
                        })
                    next_time += pattern.period
        
        return predictions
    
    async def _predict_trend(self, events: List[TemporalEvent],
                           horizon: timedelta) -> List[Dict[str, Any]]:
        """Predict based on trends"""
        predictions = []
        
        if len(events) < 5:
            return predictions
        
        # Analyze recent trend
        recent = events[-10:]
        timestamps = np.array([e.timestamp.timestamp() for e in recent])
        values = np.array([self._extract_numeric_value(e.value) for e in recent])
        
        # Fit polynomial
        z = np.polyfit(timestamps, values, 2)
        
        # Predict future values
        future_time = datetime.now() + horizon
        future_timestamp = future_time.timestamp()
        
        predicted_value = np.polyval(z, future_timestamp)
        
        # Calculate confidence based on fit quality
        residuals = values - np.polyval(z, timestamps)
        r_squared = 1 - (np.sum(residuals**2) / np.sum((values - np.mean(values))**2))
        
        if r_squared > 0.5:
            predictions.append({
                'timestamp': future_time.isoformat(),
                'predicted_value': float(predicted_value),
                'confidence': float(r_squared),
                'basis': 'trend_extrapolation'
            })
        
        return predictions
    
    async def _predict_ml(self, event_type: str, horizon: timedelta) -> List[Dict[str, Any]]:
        """ML-based prediction (placeholder for advanced models)"""
        # In a real implementation, this would use trained models
        return []
    
    async def _calculate_cross_correlation(self, events1: List[TemporalEvent],
                                         events2: List[TemporalEvent],
                                         lag_range: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Calculate cross-correlation between event streams"""
        if len(events1) < 10 or len(events2) < 10:
            return None
        
        # Create time series
        # Find common time range
        start_time = max(events1[0].timestamp, events2[0].timestamp)
        end_time = min(events1[-1].timestamp, events2[-1].timestamp)
        
        # Resample to regular intervals
        time_points = pd.date_range(start=start_time, end=end_time, freq='1H')
        
        ts1 = self._resample_events(events1, time_points)
        ts2 = self._resample_events(events2, time_points)
        
        if len(ts1) < 10 or len(ts2) < 10:
            return None
        
        # Calculate cross-correlation
        correlations = []
        lags = range(lag_range[0], lag_range[1] + 1)
        
        for lag in lags:
            if lag < 0:
                corr = np.corrcoef(ts1[:lag], ts2[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(ts1[lag:], ts2[:-lag])[0, 1]
            else:
                corr = np.corrcoef(ts1, ts2)[0, 1]
            
            correlations.append(corr)
        
        # Find maximum correlation
        max_idx = np.argmax(np.abs(correlations))
        max_corr = correlations[max_idx]
        max_lag = lags[max_idx]
        
        return {
            'event_type1': events1[0].event_type,
            'event_type2': events2[0].event_type,
            'max_correlation': float(max_corr),
            'optimal_lag': max_lag,
            'lag_unit': 'hours',
            'interpretation': self._interpret_correlation(max_corr, max_lag)
        }
    
    def _resample_events(self, events: List[TemporalEvent], 
                        time_points: pd.DatetimeIndex) -> np.ndarray:
        """Resample events to regular time intervals"""
        # Create a series from events
        event_dict = {e.timestamp: self._extract_numeric_value(e.value) for e in events}
        
        # Interpolate to regular intervals
        values = []
        for tp in time_points:
            # Find nearest event
            nearest_events = sorted(events, key=lambda e: abs((e.timestamp - tp).total_seconds()))
            if nearest_events and abs((nearest_events[0].timestamp - tp).total_seconds()) < 3600:
                values.append(self._extract_numeric_value(nearest_events[0].value))
            else:
                values.append(np.nan)
        
        # Fill NaN values
        values = pd.Series(values).fillna(method='ffill').fillna(method='bfill').values
        
        return values
    
    def _interpret_correlation(self, correlation: float, lag: int) -> str:
        """Interpret correlation results"""
        strength = abs(correlation)
        if strength > 0.8:
            strength_desc = "strong"
        elif strength > 0.5:
            strength_desc = "moderate"
        else:
            strength_desc = "weak"
        
        direction = "positive" if correlation > 0 else "negative"
        
        if lag > 0:
            timing = f"leads by {lag} hours"
        elif lag < 0:
            timing = f"lags by {-lag} hours"
        else:
            timing = "occurs simultaneously"
        
        return f"{strength_desc} {direction} correlation, first event {timing}"
    
    def _analyze_trend(self, ts: pd.Series) -> Dict[str, Any]:
        """Analyze trend in time series"""
        # Simple linear regression
        x = np.arange(len(ts))
        y = ts.values
        
        z = np.polyfit(x, y, 1)
        slope = z[0]
        
        # Determine trend
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            'trend': trend,
            'slope': float(slope),
            'change_per_hour': float(slope)
        }
    
    def _detect_seasonality(self, ts: pd.Series) -> Dict[str, Any]:
        """Detect seasonality in time series"""
        # Use FFT for frequency analysis
        fft = np.fft.fft(ts.values)
        frequencies = np.fft.fftfreq(len(ts))
        
        # Find dominant frequencies
        power = np.abs(fft)
        dominant_idx = np.argsort(power)[-5:]  # Top 5 frequencies
        
        periods = []
        for idx in dominant_idx:
            if frequencies[idx] > 0:  # Positive frequencies only
                period_hours = 1 / frequencies[idx]
                if 1 < period_hours < len(ts) / 2:  # Reasonable periods
                    periods.append({
                        'hours': float(period_hours),
                        'strength': float(power[idx] / np.mean(power))
                    })
        
        return {
            'has_seasonality': len(periods) > 0,
            'dominant_periods': sorted(periods, key=lambda p: p['strength'], reverse=True)[:3]
        }
    
    def _calculate_autocorrelation(self, ts: pd.Series) -> Dict[str, Any]:
        """Calculate autocorrelation of time series"""
        # Calculate autocorrelation for different lags
        acf_values = []
        max_lag = min(20, len(ts) // 4)
        
        for lag in range(1, max_lag + 1):
            acf = ts.autocorr(lag=lag)
            if not np.isnan(acf):
                acf_values.append({
                    'lag': lag,
                    'correlation': float(acf)
                })
        
        # Find significant lags
        significant_lags = [
            acf for acf in acf_values 
            if abs(acf['correlation']) > 0.5
        ]
        
        return {
            'significant_lags': significant_lags,
            'max_correlation': max(acf_values, key=lambda x: abs(x['correlation']))
            if acf_values else None
        }
    
    def _test_stationarity(self, ts: pd.Series) -> bool:
        """Test if time series is stationary (simplified)"""
        # Simple test: compare mean and variance of first and second half
        if len(ts) < 10:
            return True
        
        mid = len(ts) // 2
        first_half = ts[:mid]
        second_half = ts[mid:]
        
        # Check if means are similar
        mean_diff = abs(first_half.mean() - second_half.mean())
        overall_std = ts.std()
        
        # Check if variances are similar
        var_ratio = first_half.var() / (second_half.var() + 1e-6)
        
        # Simple stationarity criteria
        is_stationary = (mean_diff < overall_std * 0.5 and 
                        0.5 < var_ratio < 2.0)
        
        return is_stationary
    
    def _consolidate_patterns(self):
        """Remove duplicate or overlapping patterns"""
        # Simple deduplication based on pattern ID prefix
        unique_patterns = {}
        
        for pattern in self.detected_patterns:
            key = pattern.pattern_id.split('_')[0:2]  # Type and first identifier
            key = '_'.join(key)
            
            if key not in unique_patterns or pattern.confidence > unique_patterns[key].confidence:
                unique_patterns[key] = pattern
        
        self.detected_patterns = list(unique_patterns.values())
    
    def _extract_metric(self, event: TemporalEvent, metric: str) -> float:
        """Extract specific metric from event"""
        if metric == 'value':
            return self._extract_numeric_value(event.value)
        elif metric == 'duration' and event.duration:
            return event.duration.total_seconds()
        elif metric == 'confidence':
            return event.confidence
        else:
            return 0.0

# Create singleton instance
_temporal_system = None

async def get_temporal_system() -> TemporalProcessingSystem:
    """Get or create temporal processing system"""
    global _temporal_system
    if _temporal_system is None:
        _temporal_system = TemporalProcessingSystem()
    return _temporal_system
