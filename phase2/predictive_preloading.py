#!/usr/bin/env python3
"""
JARVIS Phase 2: Predictive Pre-loading System
Pre-loads likely next actions based on patterns and context
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from pathlib import Path
import logging
import heapq
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import psutil
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictedAction:
    """A predicted action with metadata"""
    action_id: str
    action_type: str  # app_launch, file_open, web_search, command, etc.
    target: str  # What to pre-load (app name, file path, URL, etc.)
    probability: float
    context_factors: Dict[str, float]
    pre_load_function: Optional[callable] = None
    resource_cost: float = 0.1  # Estimated resource usage (0-1)
    execution_time: float = 0.0  # Estimated time to execute
    dependencies: List[str] = field(default_factory=list)

@dataclass
class UserPattern:
    """Detected user behavior pattern"""
    pattern_id: str
    pattern_type: str  # sequential, temporal, contextual
    actions: List[str]
    frequency: int
    confidence: float
    time_windows: List[Tuple[datetime, datetime]]
    context_conditions: Dict[str, Any]

class PredictivePreloadingSystem:
    """Intelligent system for predicting and pre-loading user actions"""
    
    def __init__(self, cache_dir: str = "./jarvis_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Prediction models
        self.action_predictor = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Pattern storage
        self.user_patterns = defaultdict(list)
        self.action_history = []
        self.action_sequences = defaultdict(list)
        self.temporal_patterns = defaultdict(list)
        
        # Pre-loading management
        self.pre_loaded_resources = {}
        self.resource_cache = {}
        self.max_cache_size_mb = 500
        self.current_cache_size_mb = 0
        
        # Execution context
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.active_predictions = set()
        
        # Load existing models and patterns
        self._load_models()
        
    async def record_action(self, action_type: str, target: str, 
                          context: Dict[str, Any]) -> None:
        """Record a user action for pattern learning"""
        action = {
            'timestamp': datetime.now(),
            'type': action_type,
            'target': target,
            'context': context,
            'day_of_week': datetime.now().weekday(),
            'hour': datetime.now().hour,
            'minute': datetime.now().minute
        }
        
        self.action_history.append(action)
        
        # Update sequences
        recent_actions = [a['target'] for a in self.action_history[-5:]]
        if len(recent_actions) >= 2:
            for i in range(len(recent_actions) - 1):
                sequence = (recent_actions[i], recent_actions[i + 1])
                self.action_sequences[sequence].append(datetime.now())
        
        # Detect patterns periodically
        if len(self.action_history) % 10 == 0:
            await self._detect_patterns()
        
        # Retrain model periodically
        if len(self.action_history) % 50 == 0:
            await self._train_prediction_model()
    
    async def predict_next_actions(self, current_context: Dict[str, Any], 
                                  top_k: int = 5) -> List[PredictedAction]:
        """Predict likely next actions based on current context"""
        predictions = []
        
        # Use ML model if trained
        if self.model_trained:
            ml_predictions = await self._ml_predict(current_context)
            predictions.extend(ml_predictions)
        
        # Use pattern matching
        pattern_predictions = await self._pattern_predict(current_context)
        predictions.extend(pattern_predictions)
        
        # Use temporal patterns
        temporal_predictions = await self._temporal_predict(current_context)
        predictions.extend(temporal_predictions)
        
        # Merge and rank predictions
        merged_predictions = self._merge_predictions(predictions)
        
        # Filter by resource availability
        available_predictions = await self._filter_by_resources(merged_predictions)
        
        # Return top K
        return sorted(available_predictions, 
                     key=lambda p: p.probability, 
                     reverse=True)[:top_k]
    
    async def pre_load_resources(self, predictions: List[PredictedAction]) -> Dict[str, Any]:
        """Pre-load resources for predicted actions"""
        pre_loaded = {}
        
        for prediction in predictions:
            if prediction.probability < 0.3:  # Skip low probability
                continue
            
            if prediction.action_id in self.active_predictions:
                continue  # Already loading
            
            self.active_predictions.add(prediction.action_id)
            
            try:
                # Pre-load based on action type
                if prediction.action_type == 'app_launch':
                    result = await self._pre_load_app(prediction.target)
                elif prediction.action_type == 'file_open':
                    result = await self._pre_load_file(prediction.target)
                elif prediction.action_type == 'web_search':
                    result = await self._pre_load_search(prediction.target)
                elif prediction.action_type == 'command':
                    result = await self._pre_load_command(prediction.target)
                else:
                    result = None
                
                if result:
                    pre_loaded[prediction.action_id] = result
                    self.pre_loaded_resources[prediction.action_id] = {
                        'resource': result,
                        'timestamp': datetime.now(),
                        'prediction': prediction
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to pre-load {prediction.target}: {e}")
            finally:
                self.active_predictions.remove(prediction.action_id)
        
        # Clean old pre-loaded resources
        await self._clean_cache()
        
        return pre_loaded
    
    async def get_pre_loaded_resource(self, action_type: str, 
                                    target: str) -> Optional[Any]:
        """Get pre-loaded resource if available"""
        for resource_id, resource_data in self.pre_loaded_resources.items():
            pred = resource_data['prediction']
            if pred.action_type == action_type and pred.target == target:
                # Update access time
                resource_data['last_accessed'] = datetime.now()
                return resource_data['resource']
        
        return None
    
    async def _ml_predict(self, context: Dict[str, Any]) -> List[PredictedAction]:
        """Use ML model for predictions"""
        if not self.model_trained:
            return []
        
        # Extract features
        features = self._extract_features(context)
        
        # Get predictions
        try:
            # Get probabilities for each action
            probas = self.action_predictor.predict_proba([features])[0]
            classes = self.action_predictor.classes_
            
            predictions = []
            for i, (action_class, proba) in enumerate(zip(classes, probas)):
                if proba > 0.1:  # Threshold
                    action_type, target = action_class.split('|')
                    predictions.append(PredictedAction(
                        action_id=f"ml_{i}_{datetime.now().timestamp()}",
                        action_type=action_type,
                        target=target,
                        probability=proba,
                        context_factors={'ml_confidence': proba}
                    ))
            
            return predictions
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return []
    
    async def _pattern_predict(self, context: Dict[str, Any]) -> List[PredictedAction]:
        """Use pattern matching for predictions"""
        predictions = []
        
        if not self.action_history:
            return predictions
        
        # Get last action
        last_action = self.action_history[-1]
        last_target = last_action['target']
        
        # Find frequent sequences
        for (prev_action, next_action), timestamps in self.action_sequences.items():
            if prev_action == last_target and len(timestamps) >= 3:
                # Calculate probability based on frequency
                total_occurrences = sum(1 for a in self.action_history 
                                      if a['target'] == prev_action)
                sequence_probability = len(timestamps) / max(total_occurrences, 1)
                
                predictions.append(PredictedAction(
                    action_id=f"pattern_{prev_action}_{next_action}",
                    action_type=self._infer_action_type(next_action),
                    target=next_action,
                    probability=sequence_probability,
                    context_factors={'sequence_count': len(timestamps)}
                ))
        
        return predictions
    
    async def _temporal_predict(self, context: Dict[str, Any]) -> List[PredictedAction]:
        """Use temporal patterns for predictions"""
        predictions = []
        current_time = datetime.now()
        current_hour = current_time.hour
        current_day = current_time.weekday()
        
        # Analyze actions at similar times
        time_window_actions = defaultdict(Counter)
        
        for action in self.action_history:
            action_hour = action['timestamp'].hour
            action_day = action['timestamp'].weekday()
            
            # Same hour and day of week
            if action_hour == current_hour and action_day == current_day:
                time_window_actions['exact'][action['target']] += 1
            # Same hour any day
            elif action_hour == current_hour:
                time_window_actions['hour'][action['target']] += 1
            # Within 1 hour
            elif abs(action_hour - current_hour) <= 1:
                time_window_actions['nearby'][action['target']] += 1
        
        # Convert to predictions
        for window_type, action_counts in time_window_actions.items():
            weight = {'exact': 1.0, 'hour': 0.7, 'nearby': 0.4}[window_type]
            
            total_actions = sum(action_counts.values())
            for target, count in action_counts.most_common(5):
                probability = (count / total_actions) * weight
                
                predictions.append(PredictedAction(
                    action_id=f"temporal_{window_type}_{target}",
                    action_type=self._infer_action_type(target),
                    target=target,
                    probability=probability,
                    context_factors={
                        'temporal_pattern': window_type,
                        'occurrence_count': count
                    }
                ))
        
        return predictions
    
    async def _pre_load_app(self, app_name: str) -> Dict[str, Any]:
        """Pre-load application resources"""
        logger.info(f"Pre-loading app: {app_name}")
        
        # In a real implementation, this would:
        # 1. Load app metadata
        # 2. Pre-fetch common app data
        # 3. Warm up app caches
        
        return {
            'type': 'app',
            'name': app_name,
            'pre_loaded_at': datetime.now(),
            'metadata': {
                'icon': f"/apps/{app_name}/icon.png",
                'launch_command': f"open -a {app_name}"
            }
        }
    
    async def _pre_load_file(self, file_path: str) -> Dict[str, Any]:
        """Pre-load file contents"""
        logger.info(f"Pre-loading file: {file_path}")
        
        try:
            # Cache file metadata and first chunk
            path = Path(file_path)
            if path.exists():
                stats = path.stat()
                
                # Read first 1KB for preview
                with open(path, 'rb') as f:
                    preview = f.read(1024)
                
                return {
                    'type': 'file',
                    'path': file_path,
                    'size': stats.st_size,
                    'modified': datetime.fromtimestamp(stats.st_mtime),
                    'preview': preview,
                    'cached': True
                }
        except Exception as e:
            logger.warning(f"Failed to pre-load file {file_path}: {e}")
        
        return None
    
    async def _pre_load_search(self, query: str) -> Dict[str, Any]:
        """Pre-load search results"""
        logger.info(f"Pre-loading search: {query}")
        
        # In a real implementation, this would make actual search API calls
        # For now, return mock data
        return {
            'type': 'search',
            'query': query,
            'results_preview': [
                {'title': f'Result 1 for {query}', 'url': 'https://example.com/1'},
                {'title': f'Result 2 for {query}', 'url': 'https://example.com/2'}
            ],
            'cached_at': datetime.now()
        }
    
    async def _pre_load_command(self, command: str) -> Dict[str, Any]:
        """Pre-load command resources"""
        logger.info(f"Pre-loading command: {command}")
        
        # Pre-parse command and prepare execution environment
        return {
            'type': 'command',
            'command': command,
            'parsed': command.split(),
            'environment': dict(os.environ),
            'ready': True
        }
    
    def _merge_predictions(self, predictions: List[PredictedAction]) -> List[PredictedAction]:
        """Merge predictions from different sources"""
        # Group by target
        target_predictions = defaultdict(list)
        for pred in predictions:
            target_predictions[pred.target].append(pred)
        
        # Merge probabilities
        merged = []
        for target, preds in target_predictions.items():
            # Weighted average of probabilities
            total_prob = sum(p.probability for p in preds)
            avg_prob = total_prob / len(preds)
            
            # Boost if multiple predictors agree
            if len(preds) > 1:
                avg_prob = min(avg_prob * 1.2, 0.95)
            
            # Merge context factors
            merged_factors = {}
            for pred in preds:
                merged_factors.update(pred.context_factors)
            
            merged.append(PredictedAction(
                action_id=f"merged_{target}_{datetime.now().timestamp()}",
                action_type=preds[0].action_type,
                target=target,
                probability=avg_prob,
                context_factors=merged_factors
            ))
        
        return merged
    
    async def _filter_by_resources(self, predictions: List[PredictedAction]) -> List[PredictedAction]:
        """Filter predictions based on available resources"""
        # Check system resources
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        
        available_memory_mb = memory.available / 1024 / 1024
        
        filtered = []
        estimated_usage = 0
        
        for pred in sorted(predictions, key=lambda p: p.probability, reverse=True):
            # Estimate resource usage
            resource_cost_mb = pred.resource_cost * 100  # Rough estimate
            
            if (estimated_usage + resource_cost_mb < available_memory_mb * 0.2 and  # Use max 20% of available
                cpu < 80):  # Don't pre-load if CPU is busy
                filtered.append(pred)
                estimated_usage += resource_cost_mb
        
        return filtered
    
    async def _detect_patterns(self):
        """Detect patterns in user behavior"""
        if len(self.action_history) < 10:
            return
        
        # Sequential patterns
        sequence_counts = Counter()
        for i in range(len(self.action_history) - 2):
            seq = tuple(a['target'] for a in self.action_history[i:i+3])
            sequence_counts[seq] += 1
        
        # Create patterns from frequent sequences
        for seq, count in sequence_counts.most_common(10):
            if count >= 3:  # Minimum frequency
                pattern = UserPattern(
                    pattern_id=f"seq_{hash(seq)}",
                    pattern_type='sequential',
                    actions=list(seq),
                    frequency=count,
                    confidence=count / len(self.action_history),
                    time_windows=[],
                    context_conditions={}
                )
                self.user_patterns['sequential'].append(pattern)
    
    async def _train_prediction_model(self):
        """Train ML model on action history"""
        if len(self.action_history) < 50:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for i in range(len(self.action_history) - 1):
            features = self._extract_features({
                'timestamp': self.action_history[i]['timestamp'],
                'last_action': self.action_history[i]['target'],
                'context': self.action_history[i]['context']
            })
            
            label = f"{self.action_history[i+1]['type']}|{self.action_history[i+1]['target']}"
            
            X.append(features)
            y.append(label)
        
        try:
            # Train model
            X_scaled = self.scaler.fit_transform(X)
            self.action_predictor.fit(X_scaled, y)
            self.model_trained = True
            
            # Save model
            model_path = self.cache_dir / "prediction_model.pkl"
            joblib.dump((self.action_predictor, self.scaler), model_path)
            
            logger.info("Prediction model trained successfully")
        except Exception as e:
            logger.warning(f"Failed to train model: {e}")
    
    def _extract_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract features for ML model"""
        features = []
        
        # Time features
        if 'timestamp' in context:
            ts = context['timestamp']
            if isinstance(ts, datetime):
                features.extend([
                    ts.hour,
                    ts.minute,
                    ts.weekday(),
                    ts.day,
                    ts.month
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        else:
            features.extend([
                datetime.now().hour,
                datetime.now().minute,
                datetime.now().weekday(),
                datetime.now().day,
                datetime.now().month
            ])
        
        # Context features (simplified)
        features.append(len(context.get('last_action', '')))
        features.append(hash(context.get('last_action', '')) % 1000)
        
        # Add more features as needed
        while len(features) < 10:
            features.append(0)
        
        return features[:10]  # Ensure consistent size
    
    def _infer_action_type(self, target: str) -> str:
        """Infer action type from target"""
        if target.endswith('.app') or target in ['Safari', 'Chrome', 'Firefox']:
            return 'app_launch'
        elif '/' in target or '\\' in target or target.endswith(('.txt', '.py', '.md')):
            return 'file_open'
        elif target.startswith('http'):
            return 'web_search'
        else:
            return 'command'
    
    async def _clean_cache(self):
        """Clean old cached resources"""
        now = datetime.now()
        to_remove = []
        
        for resource_id, resource_data in self.pre_loaded_resources.items():
            # Remove if older than 5 minutes and not accessed
            age = now - resource_data['timestamp']
            last_accessed = resource_data.get('last_accessed', resource_data['timestamp'])
            time_since_access = now - last_accessed
            
            if age > timedelta(minutes=5) and time_since_access > timedelta(minutes=2):
                to_remove.append(resource_id)
        
        for resource_id in to_remove:
            del self.pre_loaded_resources[resource_id]
        
        if to_remove:
            logger.info(f"Cleaned {len(to_remove)} cached resources")
    
    def _load_models(self):
        """Load saved models if they exist"""
        model_path = self.cache_dir / "prediction_model.pkl"
        if model_path.exists():
            try:
                self.action_predictor, self.scaler = joblib.load(model_path)
                self.model_trained = True
                logger.info("Loaded prediction model")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")

# Create singleton instance
_predictive_system = None

async def get_predictive_system() -> PredictivePreloadingSystem:
    """Get or create predictive pre-loading system"""
    global _predictive_system
    if _predictive_system is None:
        _predictive_system = PredictivePreloadingSystem()
    return _predictive_system
