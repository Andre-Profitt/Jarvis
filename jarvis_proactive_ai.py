import threading
import time
import cv2
import numpy as np
from datetime import datetime, timedelta
import speech_recognition as sr
import pyttsx3
from deepface import DeepFace
import pandas as pd
import json
import asyncio
import websockets
from typing import Dict, List, Any
import os

class JARVISProactiveAssistant:
    """
    Next-generation AI assistant with real-time monitoring, 
    proactive intelligence, and continuous awareness.
    """
    
    def __init__(self):
        # Core systems
        self.voice_engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # User profile & context
        self.user_profile = {
            "name": "Andre",
            "health": {
                "lactose_intolerant": True,
                "allergies": [],
                "fitness_goals": "maintain stamina"
            },
            "schedule": {},
            "preferences": {
                "video_length": 60,  # seconds
                "speaking_pace": 150,  # words per minute
                "work_hours": {"start": 9, "end": 18}
            },
            "current_activity": None,
            "stress_baseline": 0.3,
            "performance_history": []
        }
        
        # Real-time monitoring states
        self.monitoring_active = True
        self.current_metrics = {
            "speaking_pace": 0,
            "stress_level": 0,
            "emotion": "neutral",
            "posture_score": 0,
            "engagement_score": 0,
            "last_break": datetime.now()
        }
        
        # Proactive intervention thresholds
        self.intervention_rules = {
            "stress_threshold": 0.7,
            "break_interval": 90,  # minutes
            "posture_warning": 0.6,
            "pace_variance": 0.05  # 5% variance trigger
        }
        
        # Context awareness
        self.conversation_context = []
        self.environment_context = {
            "location": "home",
            "time_of_day": None,
            "ambient_noise": 0,
            "people_present": 0
        }
        
    def start(self):
        """Initialize all monitoring systems"""
        print("🚀 JARVIS Proactive System Initializing...")
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self.voice_monitoring, daemon=True),
            threading.Thread(target=self.visual_monitoring, daemon=True),
            threading.Thread(target=self.schedule_monitoring, daemon=True),
            threading.Thread(target=self.health_monitoring, daemon=True),
            threading.Thread(target=self.performance_analysis, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
            
        # Start WebSocket server for UI
        asyncio.run(self.start_websocket_server())
        
    def voice_monitoring(self):
        """Continuous voice analysis for pace, tone, and content"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
        while self.monitoring_active:
            try:
                with self.microphone as source:
                    # Listen for speech
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # Analyze in background
                    threading.Thread(
                        target=self.analyze_speech,
                        args=(audio,),
                        daemon=True
                    ).start()
                    
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                print(f"Voice monitoring error: {e}")
                
    def analyze_speech(self, audio):
        """Analyze speech for pace, emotion, and content"""
        try:
            # Speech to text
            text = self.recognizer.recognize_google(audio)
            
            # Calculate speaking pace
            words = len(text.split())
            duration = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
            pace = (words / duration) * 60  # words per minute
            
            self.current_metrics["speaking_pace"] = pace
            
            # Check if pace is off
            target_pace = self.user_profile["preferences"]["speaking_pace"]
            variance = abs(pace - target_pace) / target_pace
            
            if variance > self.intervention_rules["pace_variance"]:
                percentage = ((pace - target_pace) / target_pace) * 100
                if percentage > 0:
                    self.intervene(f"You're {abs(percentage):.0f}% faster than rehearsal. Pacing correction recommended.")
                else:
                    self.intervene(f"You're {abs(percentage):.0f}% slower than rehearsal. Pacing correction recommended.")
                    
            # Add to context
            self.conversation_context.append({
                "text": text,
                "pace": pace,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            pass
            
    def visual_monitoring(self):
        """Monitor video for posture, emotion, and engagement"""
        cap = cv2.VideoCapture(0)
        
        while self.monitoring_active:
            ret, frame = cap.read()
            if not ret:
                continue
                
            try:
                # Emotion detection
                result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=False
                )
                
                emotion = result[0]['dominant_emotion']
                self.current_metrics["emotion"] = emotion
                
                # Check stress indicators
                stress_emotions = ['angry', 'fear', 'sad']
                if emotion in stress_emotions:
                    self.current_metrics["stress_level"] = min(
                        self.current_metrics["stress_level"] + 0.1, 1.0
                    )
                else:
                    self.current_metrics["stress_level"] = max(
                        self.current_metrics["stress_level"] - 0.05, 0
                    )
                    
                # Stress intervention
                if self.current_metrics["stress_level"] > self.intervention_rules["stress_threshold"]:
                    self.intervene(
                        "The current tone lacks emotional warmth. You're tracking as lightly stressed. "
                        "Breathe in through your nose, out through your mouth."
                    )
                    
            except Exception as e:
                pass
                
            time.sleep(1)  # Check every second
            
        cap.release()
        
    def schedule_monitoring(self):
        """Monitor calendar and provide proactive reminders"""
        while self.monitoring_active:
            now = datetime.now()
            
            # Check upcoming events
            for event_time, event in self.user_profile["schedule"].items():
                time_until = (event_time - now).total_seconds() / 60
                
                # Proactive reminders at specific intervals
                if 40 < time_until < 45:  # ~42 minutes before
                    self.intervene(f"Speaking of anticipation, you're due for {event} in {int(time_until)} minutes.")
                elif 10 < time_until < 15:
                    self.intervene(f"Reminder: {event} coming up in {int(time_until)} minutes.")
                    
            # Break reminders
            time_since_break = (now - self.current_metrics["last_break"]).total_seconds() / 60
            if time_since_break > self.intervention_rules["break_interval"]:
                self.intervene("You've been working for 90 minutes. Time for a quick break to maintain peak performance.")
                
            time.sleep(60)  # Check every minute
            
    def health_monitoring(self):
        """Monitor health-related activities and choices"""
        while self.monitoring_active:
            # This would integrate with IoT devices, smart kitchen, etc.
            # For demo, we'll simulate
            
            # Check if user is near problematic foods
            if self.detect_food_choice("cheese"):
                if self.user_profile["health"]["lactose_intolerant"]:
                    self.intervene("Also, the cheese cubes from catering, hmm, bad choice. You're still lactose intolerant.")
                    
            time.sleep(30)  # Check every 30 seconds
            
    def performance_analysis(self):
        """Analyze performance patterns and provide coaching"""
        while self.monitoring_active:
            # Analyze recent performance
            if len(self.conversation_context) > 10:
                recent = self.conversation_context[-10:]
                
                # Check for patterns
                avg_pace = np.mean([c["pace"] for c in recent])
                
                # Video length optimization
                if self.user_profile["current_activity"] == "recording":
                    self.intervene("Remember, videos under 60 seconds get 34% more engagement.")
                    
            time.sleep(20)  # Analyze every 20 seconds
            
    def intervene(self, message: str):
        """Proactively interrupt with helpful information"""
        print(f"\n🤖 JARVIS: {message}")
        
        # Voice output
        self.voice_engine.say(message)
        self.voice_engine.runAndWait()
        
        # Send to UI
        asyncio.run(self.send_to_ui({
            "type": "intervention",
            "message": message,
            "metrics": self.current_metrics,
            "timestamp": datetime.now().isoformat()
        }))
        
    def detect_food_choice(self, food_item: str) -> bool:
        """Detect if user is near specific food (would use computer vision)"""
        # Placeholder - would use real detection
        return False
        
    async def start_websocket_server(self):
        """WebSocket server for real-time UI updates"""
        async def handler(websocket, path):
            while True:
                await asyncio.sleep(1)
                await websocket.send(json.dumps({
                    "metrics": self.current_metrics,
                    "profile": self.user_profile,
                    "timestamp": datetime.now().isoformat()
                }))
                
        await websockets.serve(handler, "localhost", 8765)
        
    async def send_to_ui(self, data: Dict[str, Any]):
        """Send data to UI via WebSocket"""
        # Implementation for sending to connected clients
        pass


# Performance Coaching Module
class PerformanceCoach:
    """Analyzes and improves user performance in real-time"""
    
    def __init__(self):
        self.performance_metrics = {
            "video_recording": {
                "optimal_length": 60,
                "engagement_factors": {
                    "eye_contact": 0.8,
                    "smile_frequency": 0.3,
                    "gesture_variety": 0.6,
                    "voice_modulation": 0.7
                }
            },
            "presentation": {
                "pace_wpm": 150,
                "pause_frequency": 0.15,
                "filler_word_limit": 0.02
            }
        }
        
    def analyze_video_performance(self, video_metrics):
        """Provide specific coaching for video recording"""
        suggestions = []
        
        # Eyebrow and facial expression
        if video_metrics.get("eyebrow_lift", 0) < 0.3:
            suggestions.append("improved eyebrow lift")
            
        # Enthusiasm detection
        if video_metrics.get("enthusiasm", 0) < 0.7:
            percentage = int((0.7 - video_metrics.get("enthusiasm", 0)) * 20)
            suggestions.append(f"{percentage}% more enthusiasm")
            
        if suggestions:
            return f"Want to run it again with {' and '.join(suggestions)}?"
        else:
            return "That one was close to usable!"


# Main execution
if __name__ == "__main__":
    # Initialize JARVIS with proactive capabilities
    jarvis = JARVISProactiveAssistant()
    
    # Add some schedule items for demo
    jarvis.user_profile["schedule"] = {
        datetime.now() + timedelta(minutes=42): "paddle training",
        datetime.now() + timedelta(hours=2): "team meeting",
        datetime.now() + timedelta(hours=4): "lunch with investors"
    }
    
    print("""
    ╔══════════════════════════════════════════════════╗
    ║         JARVIS PROACTIVE AI ASSISTANT            ║
    ║                                                  ║
    ║  Features:                                       ║
    ║  • Real-time performance monitoring              ║
    ║  • Proactive interventions                       ║
    ║  • Health & schedule awareness                   ║
    ║  • Continuous coaching                           ║
    ║  • Emotion & stress detection                    ║
    ╚══════════════════════════════════════════════════╝
    """)
    
    # Start the system
    jarvis.start()
