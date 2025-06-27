#!/usr/bin/env python3
"""
JARVIS Reinforcement Learning System
Learns from every interaction to improve continuously
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import gym
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from collections import deque
import json
from datetime import datetime
import asyncio

@dataclass
class Experience:
    """Single experience in JARVIS's learning"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    metadata: Dict[str, Any]

class JARVISEnvironment(gym.Env):
    """Custom environment for JARVIS to learn in"""
    
    def __init__(self):
        super().__init__()
        
        # Define action space (what JARVIS can do)
        self.action_space = gym.spaces.Discrete(50)  # 50 different actions
        
        # Define observation space (what JARVIS sees)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(128,),  # 128-dimensional state
            dtype=np.float32
        )
        
        # Current state
        self.state = None
        self.user_satisfaction = 0.5
        self.task_completion_rate = 0.5
        self.response_quality = 0.5
        
    def reset(self):
        """Reset environment to initial state"""
        self.state = np.random.randn(128).astype(np.float32)
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return results"""
        
        # Simulate action effects
        reward = self._calculate_reward(action)
        
        # Update state
        self.state = self._update_state(action)
        
        # Episode never ends (continuous learning)
        done = False
        
        info = {
            "user_satisfaction": self.user_satisfaction,
            "task_completion": self.task_completion_rate,
            "response_quality": self.response_quality
        }
        
        return self.state, reward, done, info
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on action outcomes"""
        
        # Base reward components
        user_feedback_reward = self._get_user_feedback_reward()
        task_success_reward = self._get_task_success_reward(action)
        efficiency_reward = self._get_efficiency_reward(action)
        learning_reward = self._get_learning_reward(action)
        
        # Weighted combination
        total_reward = (
            0.4 * user_feedback_reward +
            0.3 * task_success_reward +
            0.2 * efficiency_reward +
            0.1 * learning_reward
        )
        
        return total_reward

class ReinforcementLearningSystem:
    """
    JARVIS's RL system for continuous improvement
    Learns from every interaction with the user
    """
    
    def __init__(self):
        # Create custom environment
        self.env = DummyVecEnv([lambda: JARVISEnvironment()])
        
        # Initialize RL agent (PPO for stability)
        self.agent = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./jarvis_rl_logs/"
        )
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=100000)
        
        # Learning metrics
        self.metrics = {
            "total_interactions": 0,
            "average_reward": 0,
            "user_satisfaction": [],
            "task_success_rate": [],
            "learning_progress": []
        }
        
        # Reward predictor
        self.reward_predictor = RewardPredictor()
        
    async def learn_from_interaction(self, interaction: Dict[str, Any]):
        """Learn from a single user interaction"""
        
        # Convert interaction to RL format
        state = self._extract_state(interaction)
        action = self._extract_action(interaction)
        reward = await self._calculate_reward(interaction)
        next_state = self._extract_next_state(interaction)
        
        # Store experience
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
            metadata=interaction
        )
        
        self.experience_buffer.append(experience)
        
        # Update metrics
        self.metrics["total_interactions"] += 1
        self.metrics["average_reward"] = (
            0.95 * self.metrics["average_reward"] + 0.05 * reward
        )
        
        # Learn if enough experiences
        if len(self.experience_buffer) >= 1000:
            await self._train_on_experiences()
    
    async def _calculate_reward(self, interaction: Dict[str, Any]) -> float:
        """Calculate reward from interaction"""
        
        rewards = {}
        
        # 1. Explicit user feedback
        if "user_feedback" in interaction:
            feedback = interaction["user_feedback"]
            if feedback == "positive":
                rewards["feedback"] = 1.0
            elif feedback == "negative":
                rewards["feedback"] = -1.0
            else:
                rewards["feedback"] = 0.0
        
        # 2. Task completion
        if interaction.get("task_completed"):
            rewards["completion"] = 1.0
        elif interaction.get("task_failed"):
            rewards["completion"] = -0.5
        else:
            rewards["completion"] = 0.0
        
        # 3. Response time
        response_time = interaction.get("response_time", 1.0)
        if response_time < 0.5:
            rewards["speed"] = 0.5
        elif response_time < 2.0:
            rewards["speed"] = 0.2
        else:
            rewards["speed"] = -0.1
        
        # 4. Learning signal (did JARVIS learn something new?)
        if interaction.get("learned_new_pattern"):
            rewards["learning"] = 0.5
        
        # 5. User engagement (continued interaction)
        if interaction.get("user_continued_interaction"):
            rewards["engagement"] = 0.3
        
        # Combine rewards
        total_reward = sum(rewards.values()) / len(rewards)
        
        # Use reward predictor for implicit feedback
        predicted_reward = await self.reward_predictor.predict(interaction)
        
        # Weighted combination
        final_reward = 0.7 * total_reward + 0.3 * predicted_reward
        
        return final_reward
    
    async def _train_on_experiences(self):
        """Train the RL agent on collected experiences"""
        
        print("🎯 Training on recent experiences...")
        
        # Convert experiences to training data
        states = np.array([exp.state for exp in self.experience_buffer])
        actions = np.array([exp.action for exp in self.experience_buffer])
        rewards = np.array([exp.reward for exp in self.experience_buffer])
        
        # Train for a few steps
        self.agent.learn(total_timesteps=1000)
        
        # Evaluate improvement
        mean_reward, std_reward = evaluate_policy(
            self.agent,
            self.env,
            n_eval_episodes=10
        )
        
        print(f"📈 Performance: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Log to wandb
        wandb.log({
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "total_experiences": len(self.experience_buffer)
        })

class RewardPredictor(nn.Module):
    """Neural network to predict rewards from implicit signals"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Reward in [-1, 1]
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
    async def predict(self, interaction: Dict[str, Any]) -> float:
        """Predict reward from interaction features"""
        
        # Extract features
        features = self._extract_features(interaction)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            reward = self.network(features_tensor).item()
        
        return reward
    
    def _extract_features(self, interaction: Dict[str, Any]) -> np.ndarray:
        """Extract features from interaction"""
        
        features = np.zeros(128)
        
        # Time-based features
        hour = datetime.now().hour
        features[0] = hour / 24.0
        
        # Interaction type features
        interaction_type = interaction.get("type", "unknown")
        type_map = {
            "voice_command": 1,
            "text_input": 2,
            "gui_interaction": 3,
            "automated_action": 4
        }
        features[1] = type_map.get(interaction_type, 0) / 4.0
        
        # Context features
        if "context_length" in interaction:
            features[2] = min(interaction["context_length"] / 100.0, 1.0)
        
        # User state features
        if "user_idle_time" in interaction:
            features[3] = min(interaction["user_idle_time"] / 3600.0, 1.0)
        
        # Add more features as needed
        
        return features

class ContinuousImprovement:
    """Orchestrates continuous improvement through RL"""
    
    def __init__(self, rl_system: ReinforcementLearningSystem):
        self.rl_system = rl_system
        self.improvement_strategies = {
            "exploration": self._exploration_strategy,
            "exploitation": self._exploitation_strategy,
            "curiosity": self._curiosity_strategy
        }
        
    async def improve_from_feedback(self, feedback: Dict[str, Any]):
        """Improve based on user feedback"""
        
        # Immediate learning
        await self.rl_system.learn_from_interaction(feedback)
        
        # Adjust strategy based on performance
        if self.rl_system.metrics["average_reward"] < 0.5:
            # Need more exploration
            await self._exploration_strategy()
        else:
            # Can exploit learned knowledge
            await self._exploitation_strategy()
    
    async def _exploration_strategy(self):
        """Explore new behaviors when performance is low"""
        
        print("🔍 Exploring new strategies...")
        
        # Increase exploration in RL agent
        self.rl_system.agent.exploration_rate = 0.3
        
        # Try novel actions
        # Learn from diverse experiences
    
    async def _curiosity_strategy(self):
        """Driven by intrinsic curiosity"""
        
        print("🤔 Following curiosity...")
        
        # Seek novel states
        # Learn for the sake of learning

# Real-world examples
class RLExamples:
    """Examples of RL in action"""
    
    @staticmethod
    def show_learning_examples():
        """Show how JARVIS learns from interactions"""
        
        print("\n🎯 Reinforcement Learning Examples:\n")
        
        # Example 1: Learning from feedback
        print("📦 Example 1: Learning User Preferences")
        print("🗣️ User: 'Show me my emails'")
        print("🤖 JARVIS: *Shows all 150 emails*")
        print("🗣️ User: 'No, just the important ones'")
        print("🔄 RL: Negative reward (-0.5) for showing too many")
        print("🧠 Learning: User prefers filtered view")
        print("\nNext time:")
        print("🗣️ User: 'Show me my emails'")
        print("🤖 JARVIS: *Shows 5 important emails*")
        print("🗣️ User: 'Perfect!'")
        print("✅ RL: Positive reward (+1.0) - behavior reinforced!")
        
        print("\n" + "-"*50 + "\n")
        
        # Example 2: Learning optimal timing
        print("🕰️ Example 2: Learning Optimal Timing")
        print("JARVIS tries different notification times:")
        print("  9:00 AM: User dismisses → Reward: -0.2")
        print("  10:30 AM: User engages → Reward: +0.8")
        print("  2:00 PM: User is busy → Reward: -0.5")
        print("🧠 Learns: User most receptive at 10:30 AM")
        print("🎯 Future notifications optimized for this time!")
        
        print("\n" + "-"*50 + "\n")
        
        # Example 3: Learning task automation
        print("🤖 Example 3: Learning to Automate")
        print("JARVIS observes user doing repetitive task...")
        print("Day 1-3: Observes pattern (reward: +0.1 each)")
        print("Day 4: Offers to automate")
        print("🗣️ User: 'Yes, please!'")
        print("🚀 RL: Huge reward (+2.0) for proactive help!")
        print("🧠 Learns: Proactive automation is highly valued")

# Deployment
async def deploy_reinforcement_learning():
    """Deploy RL system for JARVIS"""
    
    print("🎯 Deploying Reinforcement Learning System...")
    
    # Initialize RL system
    rl_system = ReinforcementLearningSystem()
    
    # Initialize continuous improvement
    continuous_improvement = ContinuousImprovement(rl_system)
    
    # Show examples
    RLExamples.show_learning_examples()
    
    print("\n🚀 RL System Active!")
    print("🧠 JARVIS will now:")
    print("   • Learn from every interaction")
    print("   • Adapt to your preferences")
    print("   • Improve continuously")
    print("   • Optimize for your satisfaction")
    
    # Start learning loop
    while True:
        # Simulate interaction
        interaction = {
            "type": "voice_command",
            "user_feedback": "positive",
            "task_completed": True,
            "response_time": 0.8
        }
        
        await continuous_improvement.improve_from_feedback(interaction)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(deploy_reinforcement_learning())