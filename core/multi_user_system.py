#!/usr/bin/env python3
"""
JARVIS Multi-User System
Voice recognition and personalization for multiple users.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pickle
import hashlib
from pathlib import Path

# Audio processing
import sounddevice as sd
import librosa
import speech_recognition as sr
from scipy.signal import butter, lfilter

# Machine learning
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("jarvis.multi_user")


@dataclass
class UserProfile:
    """Represents a JARVIS user profile"""
    user_id: str
    name: str
    voice_signature: Optional[np.ndarray] = None
    preferences: Dict[str, Any] = None
    command_history: List[Dict[str, Any]] = None
    created_at: datetime = None
    last_active: datetime = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {
                "wake_word": "jarvis",
                "voice_speed": 1.0,
                "voice_pitch": 1.0,
                "preferred_units": "imperial",
                "timezone": "local",
                "notifications": True,
                "privacy_mode": False
            }
        if self.command_history is None:
            self.command_history = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()


class VoiceEncoder(nn.Module):
    """Neural network for voice embedding extraction"""
    
    def __init__(self, input_size: int = 40, embedding_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_size)
        )
        
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)


class VoiceIdentification:
    """Voice-based user identification system"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.users: Dict[str, UserProfile] = {}
        self.voice_model = VoiceEncoder()
        self.gmm_models: Dict[str, GaussianMixture] = {}
        self.scaler = StandardScaler()
        self.model_path = model_path or Path.home() / ".jarvis" / "voice_models"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Audio parameters
        self.sample_rate = 16000
        self.n_mfcc = 40
        self.enrollment_duration = 10  # seconds
        self.identification_duration = 3  # seconds
        
        self.load_models()
        
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio"""
        try:
            # Pre-emphasis
            pre_emphasis = 0.97
            emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=emphasized,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            
            # Compute delta and delta-delta
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Stack features
            features = np.vstack([mfcc, delta, delta2])
            
            # Get statistics
            mean = np.mean(features, axis=1)
            std = np.std(features, axis=1)
            
            return np.concatenate([mean, std])
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.zeros(self.n_mfcc * 6)  # 3 feature types * 2 stats
            
    def enroll_user(self, name: str, audio_samples: List[np.ndarray]) -> UserProfile:
        """Enroll a new user with voice samples"""
        logger.info(f"Enrolling user: {name}")
        
        # Generate user ID
        user_id = hashlib.md5(f"{name}_{datetime.now()}".encode()).hexdigest()[:8]
        
        # Extract features from all samples
        all_features = []
        for audio in audio_samples:
            features = self.extract_features(audio)
            all_features.append(features)
            
        features_array = np.array(all_features)
        
        # Fit scaler if first user
        if len(self.users) == 0:
            self.scaler.fit(features_array)
            
        # Normalize features
        normalized_features = self.scaler.transform(features_array)
        
        # Train GMM for this user
        gmm = GaussianMixture(n_components=3, covariance_type='diag')
        gmm.fit(normalized_features)
        self.gmm_models[user_id] = gmm
        
        # Create voice embedding
        with torch.no_grad():
            features_tensor = torch.FloatTensor(normalized_features)
            embeddings = self.voice_model(features_tensor)
            voice_signature = embeddings.mean(dim=0).numpy()
            
        # Create user profile
        profile = UserProfile(
            user_id=user_id,
            name=name,
            voice_signature=voice_signature
        )
        
        self.users[user_id] = profile
        self.save_models()
        
        logger.info(f"User {name} enrolled successfully with ID: {user_id}")
        return profile
        
    def identify_user(self, audio: np.ndarray, threshold: float = 0.7) -> Optional[UserProfile]:
        """Identify user from voice sample"""
        if not self.users:
            return None
            
        # Extract features
        features = self.extract_features(audio)
        
        if len(self.users) == 1:
            # Only one user, return them if above threshold
            return list(self.users.values())[0]
            
        # Normalize features
        normalized_features = self.scaler.transform(features.reshape(1, -1))
        
        # Score against all GMM models
        scores = {}
        for user_id, gmm in self.gmm_models.items():
            score = gmm.score(normalized_features)
            scores[user_id] = score
            
        # Get best match
        if scores:
            best_user_id = max(scores, key=scores.get)
            best_score = scores[best_user_id]
            
            # Normalize scores to probability
            all_scores = np.array(list(scores.values()))
            prob = np.exp(best_score - np.max(all_scores))
            
            if prob > threshold:
                user = self.users[best_user_id]
                user.last_active = datetime.now()
                return user
                
        return None
        
    def update_user_model(self, user_id: str, audio: np.ndarray):
        """Update user's voice model with new sample"""
        if user_id not in self.users:
            return
            
        # Extract and normalize features
        features = self.extract_features(audio)
        normalized_features = self.scaler.transform(features.reshape(1, -1))
        
        # Partial fit GMM (simplified approach)
        # In production, you'd want incremental learning
        gmm = self.gmm_models[user_id]
        # This is a placeholder - real implementation would use incremental learning
        
        logger.info(f"Updated voice model for user {user_id}")
        
    def save_models(self):
        """Save voice models to disk"""
        data = {
            'users': {uid: asdict(user) for uid, user in self.users.items()},
            'gmm_models': self.gmm_models,
            'scaler': self.scaler,
            'voice_model_state': self.voice_model.state_dict()
        }
        
        with open(self.model_path / "voice_models.pkl", 'wb') as f:
            pickle.dump(data, f)
            
    def load_models(self):
        """Load voice models from disk"""
        model_file = self.model_path / "voice_models.pkl"
        if not model_file.exists():
            return
            
        try:
            with open(model_file, 'rb') as f:
                data = pickle.load(f)
                
            # Restore users
            for uid, user_data in data['users'].items():
                # Convert dict back to UserProfile
                profile = UserProfile(**user_data)
                self.users[uid] = profile
                
            self.gmm_models = data['gmm_models']
            self.scaler = data['scaler']
            self.voice_model.load_state_dict(data['voice_model_state'])
            
            logger.info(f"Loaded {len(self.users)} user profiles")
            
        except Exception as e:
            logger.error(f"Failed to load voice models: {e}")


class MultiUserManager:
    """Manages multi-user functionality for JARVIS"""
    
    def __init__(self, jarvis_instance=None):
        self.jarvis = jarvis_instance
        self.voice_id = VoiceIdentification()
        self.current_user: Optional[UserProfile] = None
        self.guest_mode = False
        self.recognizer = sr.Recognizer()
        
        # User session management
        self.session_timeout = timedelta(minutes=30)
        self.last_activity = datetime.now()
        
        # Privacy settings
        self.privacy_mode = False
        self.require_authentication = True
        
        # Load user profiles
        self.load_user_profiles()
        
    def load_user_profiles(self):
        """Load user profiles from storage"""
        profiles_path = Path.home() / ".jarvis" / "user_profiles.json"
        if profiles_path.exists():
            try:
                with open(profiles_path, 'r') as f:
                    profiles_data = json.load(f)
                    
                # Update voice ID system with loaded profiles
                for user_data in profiles_data.get('users', []):
                    user_id = user_data['user_id']
                    if user_id in self.voice_id.users:
                        # Update preferences
                        self.voice_id.users[user_id].preferences.update(
                            user_data.get('preferences', {})
                        )
                        
            except Exception as e:
                logger.error(f"Failed to load user profiles: {e}")
                
    def save_user_profiles(self):
        """Save user profiles to storage"""
        profiles_path = Path.home() / ".jarvis" / "user_profiles.json"
        profiles_path.parent.mkdir(parents=True, exist_ok=True)
        
        profiles_data = {
            'users': [
                {
                    'user_id': user.user_id,
                    'name': user.name,
                    'preferences': user.preferences,
                    'created_at': user.created_at.isoformat(),
                    'last_active': user.last_active.isoformat()
                }
                for user in self.voice_id.users.values()
            ]
        }
        
        with open(profiles_path, 'w') as f:
            json.dump(profiles_data, f, indent=2)
            
    def enroll_new_user(self, name: str) -> bool:
        """Interactive user enrollment process"""
        print(f"\n🎤 Starting voice enrollment for {name}")
        print("Please read the following phrases clearly:")
        
        enrollment_phrases = [
            "Hey JARVIS, this is my voice",
            "The quick brown fox jumps over the lazy dog",
            "JARVIS, recognize my voice pattern",
            "One, two, three, four, five, six, seven, eight, nine, zero"
        ]
        
        audio_samples = []
        
        for i, phrase in enumerate(enrollment_phrases):
            print(f"\n[{i+1}/4] Say: '{phrase}'")
            print("Press ENTER when ready, then speak...")
            input()
            
            # Record audio
            duration = 4  # seconds
            audio = sd.rec(
                int(duration * self.voice_id.sample_rate),
                samplerate=self.voice_id.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            audio_samples.append(audio.flatten())
            print("✓ Recorded")
            
        # Enroll user
        try:
            profile = self.voice_id.enroll_user(name, audio_samples)
            self.save_user_profiles()
            print(f"\n✅ Enrollment successful! User ID: {profile.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Enrollment failed: {e}")
            print(f"\n❌ Enrollment failed: {e}")
            return False
            
    def identify_speaker(self, audio: np.ndarray) -> Optional[UserProfile]:
        """Identify the current speaker"""
        user = self.voice_id.identify_user(audio)
        
        if user:
            logger.info(f"Identified user: {user.name}")
            self.current_user = user
            self.last_activity = datetime.now()
            
            # Apply user preferences
            self.apply_user_preferences(user)
            
        return user
        
    def apply_user_preferences(self, user: UserProfile):
        """Apply user-specific preferences to JARVIS"""
        if not self.jarvis:
            return
            
        prefs = user.preferences
        
        # Apply voice settings
        if hasattr(self.jarvis, 'tts_engine'):
            rate = self.jarvis.tts_engine.getProperty('rate')
            self.jarvis.tts_engine.setProperty('rate', rate * prefs.get('voice_speed', 1.0))
            
        # Apply wake word
        if hasattr(self.jarvis, 'wake_word'):
            self.jarvis.wake_word = prefs.get('wake_word', 'jarvis')
            
        # Apply privacy settings
        self.privacy_mode = prefs.get('privacy_mode', False)
        
        logger.info(f"Applied preferences for user: {user.name}")
        
    def switch_user(self, user_name: str) -> bool:
        """Manually switch to a different user"""
        for user in self.voice_id.users.values():
            if user.name.lower() == user_name.lower():
                self.current_user = user
                self.apply_user_preferences(user)
                logger.info(f"Switched to user: {user.name}")
                return True
                
        return False
        
    def handle_command(self, command: str, audio: Optional[np.ndarray] = None) -> Tuple[str, bool]:
        """Handle command with user context"""
        # Identify user if audio provided
        if audio is not None and self.require_authentication:
            user = self.identify_speaker(audio)
            if not user and not self.guest_mode:
                return "I don't recognize your voice. Please identify yourself or enable guest mode.", False
                
        # Check session timeout
        if (datetime.now() - self.last_activity) > self.session_timeout:
            self.current_user = None
            return "Session expired. Please identify yourself.", False
            
        # Update activity
        self.last_activity = datetime.now()
        
        # Log command to user history
        if self.current_user and not self.privacy_mode:
            self.current_user.command_history.append({
                'command': command,
                'timestamp': datetime.now().isoformat(),
                'success': True  # Will be updated based on actual result
            })
            
            # Limit history size
            if len(self.current_user.command_history) > 1000:
                self.current_user.command_history = self.current_user.command_history[-1000:]
                
        # Get personalized response
        response = self.get_personalized_response(command)
        
        return response, True
        
    def get_personalized_response(self, command: str) -> str:
        """Get response personalized for current user"""
        if not self.current_user:
            return f"Processing: {command}"
            
        # Personalize response
        name = self.current_user.name
        
        # Add personal touches based on history
        if "good morning" in command.lower():
            # Check if user usually asks about weather in morning
            morning_commands = [
                cmd for cmd in self.current_user.command_history
                if "morning" in cmd.get('command', '').lower()
            ]
            
            if any("weather" in cmd.get('command', '').lower() for cmd in morning_commands):
                return f"Good morning {name}! Shall I check the weather for you as usual?"
            else:
                return f"Good morning {name}! How can I help you today?"
                
        return f"Certainly, {name}. {command}"
        
    def list_users(self) -> List[Dict[str, Any]]:
        """List all enrolled users"""
        return [
            {
                'name': user.name,
                'user_id': user.user_id,
                'last_active': user.last_active.isoformat(),
                'is_active': user.is_active
            }
            for user in self.voice_id.users.values()
        ]
        
    def remove_user(self, user_id: str) -> bool:
        """Remove a user profile"""
        if user_id in self.voice_id.users:
            del self.voice_id.users[user_id]
            if user_id in self.voice_id.gmm_models:
                del self.voice_id.gmm_models[user_id]
                
            self.voice_id.save_models()
            self.save_user_profiles()
            
            logger.info(f"Removed user: {user_id}")
            return True
            
        return False
        
    def enable_guest_mode(self):
        """Enable guest mode (no authentication required)"""
        self.guest_mode = True
        self.require_authentication = False
        logger.info("Guest mode enabled")
        
    def disable_guest_mode(self):
        """Disable guest mode"""
        self.guest_mode = False
        self.require_authentication = True
        logger.info("Guest mode disabled")


class MultiUserCommandProcessor:
    """Processes multi-user specific commands"""
    
    def __init__(self, multi_user_manager: MultiUserManager):
        self.manager = multi_user_manager
        
    async def process_command(self, command: str) -> Tuple[bool, str]:
        """Process multi-user commands"""
        command_lower = command.lower()
        
        # User enrollment
        if "enroll" in command_lower or "add user" in command_lower:
            return True, "I'll help you add a new user. What's the name?"
            
        # Switch user
        if "switch to" in command_lower or "change user to" in command_lower:
            parts = command_lower.split("to")
            if len(parts) > 1:
                user_name = parts[-1].strip()
                if self.manager.switch_user(user_name):
                    return True, f"Switched to {user_name}'s profile"
                else:
                    return True, f"I don't have a profile for {user_name}"
                    
        # Who am I
        if "who am i" in command_lower or "identify me" in command_lower:
            if self.manager.current_user:
                return True, f"You are {self.manager.current_user.name}"
            else:
                return True, "I haven't identified you yet. Please say something so I can recognize your voice."
                
        # List users
        if "list users" in command_lower or "who do you know" in command_lower:
            users = self.manager.list_users()
            if users:
                names = [u['name'] for u in users]
                return True, f"I know {len(users)} users: {', '.join(names)}"
            else:
                return True, "No users enrolled yet"
                
        # Guest mode
        if "guest mode" in command_lower:
            if "enable" in command_lower or "on" in command_lower:
                self.manager.enable_guest_mode()
                return True, "Guest mode enabled. Anyone can use me now."
            elif "disable" in command_lower or "off" in command_lower:
                self.manager.disable_guest_mode()
                return True, "Guest mode disabled. Voice authentication required."
                
        # Privacy mode
        if "privacy mode" in command_lower:
            if "enable" in command_lower or "on" in command_lower:
                self.manager.privacy_mode = True
                return True, "Privacy mode enabled. I won't save your commands."
            elif "disable" in command_lower or "off" in command_lower:
                self.manager.privacy_mode = False
                return True, "Privacy mode disabled."
                
        return False, ""


def integrate_multi_user_with_jarvis(jarvis_instance) -> MultiUserManager:
    """Integrate multi-user system with JARVIS"""
    
    # Create multi-user manager
    manager = MultiUserManager(jarvis_instance)
    
    # Create command processor
    processor = MultiUserCommandProcessor(manager)
    
    # Add to JARVIS command processor if available
    if hasattr(jarvis_instance, 'command_processor'):
        # Store reference
        jarvis_instance.multi_user_manager = manager
        jarvis_instance.multi_user_processor = processor
        
        logger.info("Multi-user system integrated with JARVIS")
    
    return manager


if __name__ == "__main__":
    # Demo the multi-user system
    print("JARVIS Multi-User System Demo")
    print("=" * 40)
    
    manager = MultiUserManager()
    
    while True:
        print("\nOptions:")
        print("1. Enroll new user")
        print("2. List users")
        print("3. Test voice identification")
        print("4. Enable guest mode")
        print("5. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            name = input("Enter user name: ").strip()
            manager.enroll_new_user(name)
            
        elif choice == "2":
            users = manager.list_users()
            if users:
                print("\nEnrolled users:")
                for user in users:
                    print(f"- {user['name']} (ID: {user['user_id']})")
            else:
                print("No users enrolled")
                
        elif choice == "3":
            print("Say something...")
            # Record audio
            duration = 3
            audio = sd.rec(
                int(duration * 16000),
                samplerate=16000,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            user = manager.identify_speaker(audio.flatten())
            if user:
                print(f"Identified: {user.name}")
            else:
                print("Could not identify speaker")
                
        elif choice == "4":
            manager.enable_guest_mode()
            print("Guest mode enabled")
            
        elif choice == "5":
            break