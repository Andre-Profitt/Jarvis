"""
JARVIS Phase 6: Natural Language Flow & Emotional Continuity
==========================================================
Implements context-aware conversational intelligence with emotional persistence
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import re
import random

class EmotionalTone(Enum):
    """Emotional tones JARVIS can adopt"""
    SUPPORTIVE = "supportive"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    URGENT = "urgent"
    EMPATHETIC = "empathetic"
    MOTIVATIONAL = "motivational"
    CALM = "calm"
    EXCITED = "excited"

class ConversationTopic(Enum):
    """Topics to track context switching"""
    WORK = "work"
    HEALTH = "health"
    PERSONAL = "personal"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    PLANNING = "planning"
    CRISIS = "crisis"
    CASUAL = "casual"

@dataclass
class ConversationContext:
    """Maintains conversation state and history"""
    topic: ConversationTopic = ConversationTopic.CASUAL
    emotional_tone: EmotionalTone = EmotionalTone.CASUAL
    entities: Dict[str, Any] = field(default_factory=dict)
    recent_intents: deque = field(default_factory=lambda: deque(maxlen=10))
    active_threads: Dict[str, Dict] = field(default_factory=dict)
    interrupt_stack: List[Dict] = field(default_factory=list)
    last_interaction: datetime = field(default_factory=datetime.now)
    mood_trajectory: deque = field(default_factory=lambda: deque(maxlen=20))
    
class NaturalLanguageFlow:
    """Manages natural, context-aware conversations with emotional continuity"""
    
    def __init__(self):
        self.context = ConversationContext()
        self.emotional_memory = deque(maxlen=50)  # Remember emotional states
        self.topic_history = deque(maxlen=30)     # Track topic transitions
        self.response_patterns = self._initialize_response_patterns()
        self.continuity_phrases = self._initialize_continuity_phrases()
        self.interrupt_handlers = self._initialize_interrupt_handlers()
        
    def _initialize_response_patterns(self) -> Dict:
        """Define natural response patterns for different contexts"""
        return {
            EmotionalTone.SUPPORTIVE: {
                "acknowledgment": ["I understand", "That makes sense", "I hear you"],
                "encouragement": ["You're doing great", "Keep it up", "That's progress"],
                "validation": ["Your feelings are valid", "It's okay to feel that way"],
            },
            EmotionalTone.PROFESSIONAL: {
                "acknowledgment": ["Understood", "Noted", "I'll handle that"],
                "status": ["Processing", "On it", "Working on that now"],
                "completion": ["Done", "Completed", "Finished"],
            },
            EmotionalTone.CASUAL: {
                "greeting": ["Hey there", "What's up", "How's it going"],
                "acknowledgment": ["Got it", "Sure thing", "No problem"],
                "farewell": ["Catch you later", "Take care", "See ya"],
            },
            EmotionalTone.EMPATHETIC: {
                "understanding": ["I can see why that would be difficult", "That sounds challenging"],
                "support": ["I'm here for you", "We'll work through this together"],
                "validation": ["Your reaction is completely understandable"],
            }
        }
        
    def _initialize_continuity_phrases(self) -> Dict:
        """Phrases to maintain conversation continuity"""
        return {
            "topic_return": [
                "Getting back to what we were discussing about {topic}...",
                "So, regarding {topic} we mentioned earlier...",
                "Circling back to {topic}...",
            ],
            "context_bridge": [
                "Speaking of which...",
                "That reminds me...",
                "On a related note...",
            ],
            "interrupt_acknowledge": [
                "I'll note that - but first, about {current_topic}...",
                "Good point, let me finish this thought and we'll come back to that...",
                "I heard you - give me just a moment to wrap this up...",
            ],
            "emotional_transition": [
                "I sense the mood has shifted...",
                "I notice you seem {emotion} now...",
                "Your energy feels different...",
            ]
        }
        
    def _initialize_interrupt_handlers(self) -> Dict:
        """Define how to handle conversation interrupts"""
        return {
            "urgent": self._handle_urgent_interrupt,
            "question": self._handle_question_interrupt,
            "topic_change": self._handle_topic_change_interrupt,
            "emotional": self._handle_emotional_interrupt,
        }
        
    async def process_input(self, text: str, metadata: Dict = None) -> Dict:
        """Process input with context awareness and emotional continuity"""
        # Detect conversation dynamics
        interrupt_type = self._detect_interrupt(text, metadata)
        emotion = self._analyze_emotion(text, metadata)
        topic = self._identify_topic(text, self.context.topic)
        intent = self._extract_intent(text)
        
        # Update context
        self._update_context(topic, emotion, intent, text)
        
        # Handle interrupts if detected
        if interrupt_type:
            response = await self._handle_interrupt(interrupt_type, text, metadata)
        else:
            response = await self._generate_contextual_response(text, metadata)
            
        # Maintain emotional continuity
        response = self._apply_emotional_continuity(response, emotion)
        
        # Add natural flow elements
        response = self._add_conversational_flow(response)
        
        return {
            "response": response,
            "context": self._serialize_context(),
            "emotional_tone": self.context.emotional_tone.value,
            "topic": self.context.topic.value,
            "continuity_score": self._calculate_continuity_score(),
        }
        
    def _detect_interrupt(self, text: str, metadata: Dict) -> Optional[str]:
        """Detect if user is interrupting or changing context"""
        # Check for explicit interruption phrases
        interrupt_phrases = ["wait", "actually", "hang on", "stop", "nevermind", "but first"]
        for phrase in interrupt_phrases:
            if phrase in text.lower():
                return "urgent" if any(word in text.lower() for word in ["emergency", "urgent", "help"]) else "topic_change"
                
        # Check for sudden topic change
        if metadata and metadata.get("typing_speed", 0) > 150:  # Fast typing might indicate urgency
            return "urgent"
            
        # Check for question in middle of JARVIS response
        if metadata and metadata.get("jarvis_speaking", False) and "?" in text:
            return "question"
            
        # Check for emotional shift
        if self._detect_emotional_shift(text, metadata):
            return "emotional"
            
        return None
        
    def _analyze_emotion(self, text: str, metadata: Dict) -> EmotionalTone:
        """Analyze emotional content of input"""
        text_lower = text.lower()
        
        # Crisis indicators
        crisis_words = ["panic", "emergency", "help", "crisis", "urgent", "immediately"]
        if any(word in text_lower for word in crisis_words):
            return EmotionalTone.URGENT
            
        # Stress indicators
        stress_words = ["stressed", "overwhelmed", "anxious", "worried", "frustrated"]
        if any(word in text_lower for word in stress_words):
            return EmotionalTone.EMPATHETIC
            
        # Positive indicators
        positive_words = ["great", "awesome", "excited", "happy", "good"]
        if any(word in text_lower for word in positive_words):
            return EmotionalTone.EXCITED
            
        # Professional context
        work_words = ["meeting", "deadline", "project", "report", "task"]
        if any(word in text_lower for word in work_words):
            return EmotionalTone.PROFESSIONAL
            
        # Check metadata for biometric indicators
        if metadata:
            if metadata.get("heart_rate", 70) > 100:
                return EmotionalTone.CALM  # User needs calming
            if metadata.get("stress_level", 0) > 0.7:
                return EmotionalTone.SUPPORTIVE
                
        return EmotionalTone.CASUAL
        
    def _identify_topic(self, text: str, current_topic: ConversationTopic) -> ConversationTopic:
        """Identify conversation topic with context awareness"""
        text_lower = text.lower()
        
        topic_keywords = {
            ConversationTopic.WORK: ["meeting", "project", "deadline", "task", "email", "report"],
            ConversationTopic.HEALTH: ["workout", "exercise", "sleep", "tired", "energy", "health"],
            ConversationTopic.PERSONAL: ["family", "friend", "weekend", "plans", "home"],
            ConversationTopic.TECHNICAL: ["code", "bug", "feature", "system", "error", "debug"],
            ConversationTopic.CREATIVE: ["idea", "design", "create", "imagine", "art"],
            ConversationTopic.PLANNING: ["schedule", "plan", "tomorrow", "week", "organize"],
            ConversationTopic.CRISIS: ["emergency", "urgent", "help", "critical"],
        }
        
        # Check for topic keywords
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
                
        if detected_topics:
            # If crisis detected, prioritize it
            if ConversationTopic.CRISIS in detected_topics:
                return ConversationTopic.CRISIS
            # Otherwise return first detected or stick with current if multiple
            return detected_topics[0]
            
        # Maintain current topic if no strong indicators
        return current_topic
        
    def _extract_intent(self, text: str) -> str:
        """Extract user intent from text"""
        text_lower = text.lower()
        
        # Question patterns
        if any(text_lower.startswith(q) for q in ["what", "when", "where", "why", "how", "can you", "could you"]):
            return "question"
        elif "?" in text:
            return "question"
            
        # Command patterns
        command_verbs = ["show", "tell", "give", "find", "create", "make", "start", "stop", "cancel"]
        if any(text_lower.startswith(verb) for verb in command_verbs):
            return "command"
            
        # Statement patterns
        if any(phrase in text_lower for phrase in ["i think", "i feel", "i want", "i need"]):
            return "statement"
            
        # Confirmation patterns
        if text_lower in ["yes", "no", "okay", "sure", "alright", "fine"]:
            return "confirmation"
            
        return "general"
        
    def _update_context(self, topic: ConversationTopic, emotion: EmotionalTone, 
                       intent: str, text: str):
        """Update conversation context with new information"""
        # Update topic if changed
        if topic != self.context.topic:
            self.topic_history.append({
                "from": self.context.topic,
                "to": topic,
                "timestamp": datetime.now(),
                "trigger": text[:50]
            })
            self.context.topic = topic
            
        # Update emotional tone with smoothing
        self.emotional_memory.append(emotion)
        self.context.emotional_tone = self._calculate_emotional_continuity()
        
        # Track intent
        self.context.recent_intents.append({
            "intent": intent,
            "timestamp": datetime.now(),
            "text": text[:100]
        })
        
        # Extract and update entities (people, places, things mentioned)
        entities = self._extract_entities(text)
        self.context.entities.update(entities)
        
        # Update mood trajectory
        self.context.mood_trajectory.append({
            "emotion": emotion,
            "timestamp": datetime.now()
        })
        
        self.context.last_interaction = datetime.now()
        
    def _extract_entities(self, text: str) -> Dict:
        """Extract named entities from text"""
        entities = {}
        
        # Simple pattern matching for common entities
        # Time entities
        time_patterns = [
            (r'\b(\d{1,2}:\d{2})\b', 'time'),
            (r'\b(today|tomorrow|yesterday)\b', 'relative_date'),
            (r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 'day'),
        ]
        
        # People (capitalized words not at sentence start)
        words = text.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and word.lower() not in ['i', 'jarvis']:
                entities[word] = 'person'
                
        # Apply patterns
        for pattern, entity_type in time_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                entities[match] = entity_type
                
        return entities
        
    def _calculate_emotional_continuity(self) -> EmotionalTone:
        """Calculate appropriate emotional tone based on history"""
        if not self.emotional_memory:
            return EmotionalTone.CASUAL
            
        # Get recent emotional tones
        recent_emotions = list(self.emotional_memory)[-5:]
        
        # If crisis detected recently, maintain urgency
        if EmotionalTone.URGENT in recent_emotions:
            return EmotionalTone.URGENT
            
        # Calculate most common recent emotion
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        # Return most common, with slight bias toward supportive tones
        max_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Smooth transitions
        if max_emotion == EmotionalTone.URGENT and emotion_counts.get(EmotionalTone.CALM, 0) > 0:
            return EmotionalTone.SUPPORTIVE
            
        return max_emotion
        
    async def _handle_interrupt(self, interrupt_type: str, text: str, metadata: Dict) -> str:
        """Handle conversation interrupts gracefully"""
        handler = self.interrupt_handlers.get(interrupt_type)
        if handler:
            return await handler(text, metadata)
        return await self._handle_generic_interrupt(text, metadata)
        
    async def _handle_urgent_interrupt(self, text: str, metadata: Dict) -> str:
        """Handle urgent interruptions"""
        # Save current context
        self.context.interrupt_stack.append({
            "topic": self.context.topic,
            "thread": self.context.active_threads.copy(),
            "timestamp": datetime.now()
        })
        
        # Switch to crisis mode
        self.context.topic = ConversationTopic.CRISIS
        self.context.emotional_tone = EmotionalTone.URGENT
        
        return "I'm stopping everything else. What's the emergency?"
        
    async def _handle_question_interrupt(self, text: str, metadata: Dict) -> str:
        """Handle mid-conversation questions"""
        # Acknowledge the question while maintaining thread
        responses = [
            "Quick answer to your question: {answer}. Now, back to {topic}...",
            "Let me address that briefly: {answer}. Continuing with {topic}...",
            "{answer}. But as I was saying about {topic}...",
        ]
        
        # Generate quick answer
        answer = await self._generate_quick_answer(text)
        response = random.choice(responses).format(
            answer=answer,
            topic=self.context.topic.value
        )
        
        return response
        
    async def _handle_topic_change_interrupt(self, text: str, metadata: Dict) -> str:
        """Handle topic changes smoothly"""
        # Acknowledge the shift
        old_topic = self.context.topic
        
        # Save current thread if valuable
        if self.context.active_threads:
            thread_id = f"thread_{datetime.now().timestamp()}"
            self.context.interrupt_stack.append({
                "id": thread_id,
                "topic": old_topic,
                "threads": self.context.active_threads.copy(),
                "can_resume": True
            })
            
        acknowledgments = [
            "Sure, let's shift gears to {new_topic}.",
            "No problem, switching to {new_topic}.",
            "Got it, let's talk about {new_topic} instead.",
        ]
        
        new_topic = self._identify_topic(text, ConversationTopic.CASUAL)
        response = random.choice(acknowledgments).format(new_topic=new_topic.value)
        
        # Add resumption hint if thread was important
        if old_topic in [ConversationTopic.WORK, ConversationTopic.PLANNING]:
            response += " We can come back to the {} discussion later if needed.".format(old_topic.value)
            
        return response
        
    async def _handle_emotional_interrupt(self, text: str, metadata: Dict) -> str:
        """Handle emotional shifts with empathy"""
        emotion = self._analyze_emotion(text, metadata)
        
        if emotion == EmotionalTone.URGENT:
            return "I sense something's really bothering you. What's going on?"
        elif emotion == EmotionalTone.EMPATHETIC:
            return "I notice you seem stressed. Want to talk about it?"
        else:
            return "I feel the shift in your mood. How are you doing?"
            
    async def _handle_generic_interrupt(self, text: str, metadata: Dict) -> str:
        """Generic interrupt handling"""
        return "I hear you. What would you like to focus on?"
        
    async def _generate_contextual_response(self, text: str, metadata: Dict) -> str:
        """Generate response with full context awareness"""
        intent = self.context.recent_intents[-1]["intent"] if self.context.recent_intents else "general"
        
        # Build response based on intent and context
        if intent == "question":
            response = await self._answer_question(text, metadata)
        elif intent == "command":
            response = await self._execute_command(text, metadata)
        elif intent == "statement":
            response = await self._respond_to_statement(text, metadata)
        else:
            response = await self._general_response(text, metadata)
            
        # Add context from previous threads if relevant
        if self._should_reference_previous_context(text):
            response = self._add_context_reference(response)
            
        return response
        
    async def _answer_question(self, text: str, metadata: Dict) -> str:
        """Answer questions with context awareness"""
        # Check if question relates to previous topics
        relevant_history = self._find_relevant_history(text)
        
        base_answer = f"Based on our discussion, "  # Placeholder for actual answer
        
        if relevant_history:
            base_answer = f"As we discussed earlier, " + base_answer
            
        return base_answer
        
    async def _execute_command(self, text: str, metadata: Dict) -> str:
        """Execute commands with appropriate tone"""
        tone_responses = {
            EmotionalTone.PROFESSIONAL: "Executing that now.",
            EmotionalTone.CASUAL: "On it!",
            EmotionalTone.SUPPORTIVE: "I'll take care of that for you.",
            EmotionalTone.URGENT: "Doing that immediately.",
        }
        
        return tone_responses.get(self.context.emotional_tone, "Working on that.")
        
    async def _respond_to_statement(self, text: str, metadata: Dict) -> str:
        """Respond to statements with emotional intelligence"""
        # Analyze statement type
        if "i feel" in text.lower():
            return self._respond_to_feeling(text)
        elif "i think" in text.lower():
            return self._respond_to_thought(text)
        elif "i need" in text.lower():
            return self._respond_to_need(text)
        else:
            return self._acknowledge_statement(text)
            
    def _respond_to_feeling(self, text: str) -> str:
        """Respond to emotional statements"""
        responses = self.response_patterns[EmotionalTone.EMPATHETIC]
        return random.choice(responses["validation"])
        
    def _respond_to_thought(self, text: str) -> str:
        """Respond to thoughts/opinions"""
        return "That's an interesting perspective."
        
    def _respond_to_need(self, text: str) -> str:
        """Respond to stated needs"""
        return "I understand what you need. Let me help with that."
        
    def _acknowledge_statement(self, text: str) -> str:
        """General acknowledgment"""
        tone = self.context.emotional_tone
        responses = self.response_patterns.get(tone, self.response_patterns[EmotionalTone.CASUAL])
        return random.choice(responses["acknowledgment"])
        
    async def _general_response(self, text: str, metadata: Dict) -> str:
        """Generate general contextual response"""
        return "I understand. Tell me more."
        
    async def _generate_quick_answer(self, text: str) -> str:
        """Generate quick answer for interrupting questions"""
        # Simplified for example
        return "Here's a quick answer"
        
    def _should_reference_previous_context(self, text: str) -> bool:
        """Determine if we should reference previous conversation threads"""
        # Check if enough time has passed
        if self.context.last_interaction:
            time_gap = datetime.now() - self.context.last_interaction
            if time_gap > timedelta(minutes=5):
                return True
                
        # Check if user references previous context
        context_words = ["earlier", "before", "mentioned", "said", "discussed"]
        return any(word in text.lower() for word in context_words)
        
    def _add_context_reference(self, response: str) -> str:
        """Add reference to previous context"""
        if self.context.interrupt_stack:
            last_context = self.context.interrupt_stack[-1]
            if last_context.get("can_resume"):
                response += f" By the way, we can return to our {last_context['topic'].value} discussion whenever you're ready."
        return response
        
    def _find_relevant_history(self, text: str) -> Optional[Dict]:
        """Find relevant conversation history"""
        # Search entities
        for entity in self.context.entities:
            if entity.lower() in text.lower():
                return {"entity": entity, "type": "referenced"}
                
        # Search recent topics
        text_lower = text.lower()
        for topic_change in self.topic_history:
            if topic_change["to"].value in text_lower:
                return {"topic": topic_change["to"], "type": "topic_reference"}
                
        return None
        
    def _detect_emotional_shift(self, text: str, metadata: Dict) -> bool:
        """Detect significant emotional shifts"""
        current_emotion = self._analyze_emotion(text, metadata)
        
        if len(self.emotional_memory) < 2:
            return False
            
        recent_emotions = list(self.emotional_memory)[-3:]
        
        # Check for sudden shift
        if all(e != current_emotion for e in recent_emotions):
            # Significant shift if moving between emotion categories
            calm_emotions = [EmotionalTone.CASUAL, EmotionalTone.PROFESSIONAL, EmotionalTone.CALM]
            active_emotions = [EmotionalTone.URGENT, EmotionalTone.EXCITED]
            
            was_calm = all(e in calm_emotions for e in recent_emotions)
            is_active = current_emotion in active_emotions
            
            return was_calm and is_active
            
        return False
        
    def _apply_emotional_continuity(self, response: str, current_emotion: EmotionalTone) -> str:
        """Apply emotional continuity to response"""
        # If emotion has shifted significantly, acknowledge it
        if self._detect_emotional_shift("", {"emotion": current_emotion}):
            transition = random.choice(self.continuity_phrases["emotional_transition"])
            response = transition.format(emotion=current_emotion.value) + " " + response
            
        return response
        
    def _add_conversational_flow(self, response: str) -> str:
        """Add natural conversational flow elements"""
        # Add continuity phrases when appropriate
        if self.context.interrupt_stack and random.random() < 0.3:
            bridge = random.choice(self.continuity_phrases["context_bridge"])
            response = bridge + " " + response
            
        # Add natural endings based on tone
        if self.context.emotional_tone == EmotionalTone.SUPPORTIVE:
            if not response.endswith("?"):
                response += " How does that sound?"
        elif self.context.emotional_tone == EmotionalTone.CASUAL:
            if random.random() < 0.2:
                response += " 😊"
                
        return response
        
    def _calculate_continuity_score(self) -> float:
        """Calculate how well conversation continuity is maintained"""
        score = 1.0
        
        # Penalize frequent topic changes
        recent_changes = [t for t in self.topic_history if 
                         (datetime.now() - t["timestamp"]).seconds < 300]
        score -= len(recent_changes) * 0.1
        
        # Reward emotional stability
        if len(set(list(self.emotional_memory)[-5:])) == 1:
            score += 0.2
            
        # Reward context references
        if self.context.entities:
            score += 0.1
            
        return max(0.0, min(1.0, score))
        
    def _serialize_context(self) -> Dict:
        """Serialize context for storage/transmission"""
        return {
            "topic": self.context.topic.value,
            "emotional_tone": self.context.emotional_tone.value,
            "entities": self.context.entities,
            "active_threads": len(self.context.active_threads),
            "interrupt_stack": len(self.context.interrupt_stack),
            "mood_trajectory": [
                {"emotion": m["emotion"].value, "time": m["timestamp"].isoformat()}
                for m in list(self.context.mood_trajectory)[-5:]
            ]
        }
        
    async def resume_context(self, context_id: str) -> str:
        """Resume a previous conversation thread"""
        for saved_context in self.context.interrupt_stack:
            if saved_context.get("id") == context_id:
                self.context.topic = saved_context["topic"]
                self.context.active_threads = saved_context["threads"]
                
                resume_phrase = random.choice(self.continuity_phrases["topic_return"])
                return resume_phrase.format(topic=saved_context["topic"].value)
                
        return "I don't have that conversation thread saved anymore."
        
    def get_conversation_summary(self) -> Dict:
        """Get summary of conversation dynamics"""
        return {
            "duration": (datetime.now() - self.context.last_interaction).seconds,
            "topics_covered": list(set(t["to"].value for t in self.topic_history)),
            "emotional_journey": [e.value for e in list(self.emotional_memory)[-10:]],
            "continuity_score": self._calculate_continuity_score(),
            "entities_discussed": list(self.context.entities.keys()),
            "interrupts_handled": len(self.context.interrupt_stack),
        }
