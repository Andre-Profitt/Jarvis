#!/usr/bin/env python3
"""
World-Class Multi-Agent Swarm System for JARVIS
Implements state-of-the-art swarm intelligence and coordination
"""

import asyncio
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Try to import Ray for distributed computing
try:
    import ray
    RAY_AVAILABLE = True
    logger.info("Ray is available for distributed computing")
except ImportError:
    ray = None
    RAY_AVAILABLE = False
    logger.info("Ray not available - using local execution only")
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
import networkx as nx
import json
import logging
from datetime import datetime
import hashlib
import random
from collections import defaultdict, deque
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import aioredis

# Logging is already configured at the top of the file

# Advanced Communication Protocols
class ACLMessage:
    """FIPA-ACL compliant message structure"""
    
    class Performative(Enum):
        INFORM = "inform"
        REQUEST = "request"
        PROPOSE = "propose"
        ACCEPT_PROPOSAL = "accept-proposal"
        REJECT_PROPOSAL = "reject-proposal"
        CFP = "call-for-proposal"
        AGREE = "agree"
        REFUSE = "refuse"
        FAILURE = "failure"
        SUBSCRIBE = "subscribe"
        QUERY = "query"
        
    def __init__(self, 
                 performative: Performative,
                 sender: str,
                 receiver: str,
                 content: Any,
                 conversation_id: str = None,
                 reply_to: str = None,
                 ontology: str = "jarvis-swarm",
                 language: str = "json"):
        self.performative = performative
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.conversation_id = conversation_id or hashlib.md5(
            f"{sender}{receiver}{datetime.now()}".encode()
        ).hexdigest()
        self.reply_to = reply_to
        self.ontology = ontology
        self.language = language
        self.timestamp = datetime.now()


@dataclass
class SwarmTask:
    """Enhanced task structure for swarm processing"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: float = 1.0
    deadline: Optional[datetime] = None
    required_capabilities: Set[str] = field(default_factory=set)
    estimated_complexity: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    reward: float = 1.0
    status: str = "pending"
    assigned_agents: List[str] = field(default_factory=list)
    results: Optional[Any] = None


class SwarmBehavior(ABC):
    """Abstract base for swarm behaviors"""
    
    @abstractmethod
    async def execute(self, agent: 'SwarmAgent', neighbors: List['SwarmAgent']) -> Any:
        pass


class FlockingBehavior(SwarmBehavior):
    """Reynolds flocking rules implementation"""
    
    def __init__(self, 
                 separation_weight: float = 1.5,
                 alignment_weight: float = 1.0,
                 cohesion_weight: float = 1.0):
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
    
    async def execute(self, agent: 'SwarmAgent', neighbors: List['SwarmAgent']) -> np.ndarray:
        if not neighbors:
            return np.zeros(3)
        
        # Separation: steer to avoid crowding
        separation = np.zeros(3)
        for neighbor in neighbors:
            diff = agent.position - neighbor.position
            distance = np.linalg.norm(diff)
            if distance > 0 and distance < 2.0:
                separation += diff / distance
        
        # Alignment: steer towards average heading
        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        alignment = avg_velocity - agent.velocity
        
        # Cohesion: steer towards average position
        avg_position = np.mean([n.position for n in neighbors], axis=0)
        cohesion = avg_position - agent.position
        
        # Combine behaviors
        steering = (
            self.separation_weight * separation +
            self.alignment_weight * alignment +
            self.cohesion_weight * cohesion
        )
        
        return steering


class AntColonyOptimization(SwarmBehavior):
    """Ant Colony Optimization for pathfinding"""
    
    def __init__(self, 
                 pheromone_deposit: float = 1.0,
                 evaporation_rate: float = 0.1,
                 alpha: float = 1.0,  # pheromone importance
                 beta: float = 2.0):  # heuristic importance
        self.pheromone_deposit = pheromone_deposit
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromone_trails = defaultdict(float)
    
    async def execute(self, agent: 'SwarmAgent', neighbors: List['SwarmAgent']) -> Dict[str, Any]:
        # Update pheromone trails
        for edge in agent.path_history:
            self.pheromone_trails[edge] *= (1 - self.evaporation_rate)
            self.pheromone_trails[edge] += self.pheromone_deposit
        
        # Choose next move based on pheromone and heuristic
        available_moves = agent.get_available_moves()
        probabilities = []
        
        for move in available_moves:
            pheromone = self.pheromone_trails.get((agent.position, move), 0.1)
            heuristic = 1.0 / (agent.distance_to(move) + 0.1)
            probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(probability)
        
        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
            chosen_idx = np.random.choice(len(available_moves), p=probabilities)
            return {"next_move": available_moves[chosen_idx]}
        
        return {"next_move": random.choice(available_moves)}


class ParticleSwarmOptimization(SwarmBehavior):
    """PSO for optimization problems"""
    
    def __init__(self,
                 inertia_weight: float = 0.7,
                 cognitive_weight: float = 1.4,
                 social_weight: float = 1.4):
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
    
    async def execute(self, agent: 'SwarmAgent', neighbors: List['SwarmAgent']) -> np.ndarray:
        # Get personal best and global best
        personal_best = agent.personal_best_position
        global_best = max(neighbors, key=lambda n: n.best_fitness).personal_best_position
        
        # Calculate velocity update
        r1, r2 = np.random.random(2)
        
        cognitive_component = self.cognitive_weight * r1 * (personal_best - agent.position)
        social_component = self.social_weight * r2 * (global_best - agent.position)
        
        new_velocity = (
            self.inertia_weight * agent.velocity +
            cognitive_component +
            social_component
        )
        
        return new_velocity


class SwarmAgent:
    """Advanced autonomous agent with swarm intelligence"""
    
    def __init__(self, 
                 agent_id: str,
                 agent_type: str,
                 capabilities: Set[str],
                 position: Optional[np.ndarray] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.position = position or np.random.randn(3)
        self.velocity = np.zeros(3)
        
        # Communication
        self.inbox = asyncio.Queue()
        self.conversations = {}
        self.subscriptions = defaultdict(list)
        
        # Learning and memory
        self.knowledge_base = {}
        self.experience_buffer = deque(maxlen=1000)
        self.personal_best_position = self.position.copy()
        self.best_fitness = -float('inf')
        
        # Swarm behaviors
        self.behaviors = {
            'flocking': FlockingBehavior(),
            'aco': AntColonyOptimization(),
            'pso': ParticleSwarmOptimization()
        }
        
        # Task management
        self.current_tasks = []
        self.task_history = []
        self.reputation = 1.0
        
        # Path history for ACO
        self.path_history = []
        
        # Neural decision network
        self.decision_network = self._build_decision_network()
        
    def _build_decision_network(self) -> nn.Module:
        """Build neural network for decision making"""
        
        class DecisionNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.Softmax(dim=-1)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return DecisionNet()
    
    async def receive_message(self, message: ACLMessage):
        """Process incoming ACL messages"""
        
        await self.inbox.put(message)
        
        # Handle different performatives
        if message.performative == ACLMessage.Performative.CFP:
            await self._handle_cfp(message)
        elif message.performative == ACLMessage.Performative.PROPOSE:
            await self._handle_proposal(message)
        elif message.performative == ACLMessage.Performative.SUBSCRIBE:
            await self._handle_subscription(message)
        elif message.performative == ACLMessage.Performative.INFORM:
            await self._handle_inform(message)
    
    async def _handle_cfp(self, message: ACLMessage):
        """Handle Call for Proposal"""
        
        task = message.content
        
        # Evaluate if we can handle the task
        can_handle = all(cap in self.capabilities for cap in task.required_capabilities)
        
        if can_handle:
            # Calculate bid based on current load and capability match
            current_load = len(self.current_tasks)
            capability_match = len(task.required_capabilities.intersection(self.capabilities))
            
            bid = {
                'agent_id': self.agent_id,
                'price': task.reward / (capability_match * self.reputation),
                'estimated_time': task.estimated_complexity / (1 + capability_match),
                'confidence': min(0.95, self.reputation * capability_match / len(task.required_capabilities))
            }
            
            # Send proposal
            proposal = ACLMessage(
                performative=ACLMessage.Performative.PROPOSE,
                sender=self.agent_id,
                receiver=message.sender,
                content=bid,
                conversation_id=message.conversation_id
            )
            
            # This would send to the actual agent
            logger.info(f"Agent {self.agent_id} proposing bid: {bid}")
    
    async def _handle_proposal(self, message: ACLMessage):
        """Handle incoming proposals"""
        
        proposal = message.content
        
        # Evaluate proposal using decision network
        features = self._extract_proposal_features(proposal)
        decision_input = torch.tensor(features, dtype=torch.float32)
        decision = self.decision_network(decision_input)
        
        if decision.argmax() == 0:  # Accept
            response = ACLMessage(
                performative=ACLMessage.Performative.ACCEPT_PROPOSAL,
                sender=self.agent_id,
                receiver=message.sender,
                content={"accepted": True},
                conversation_id=message.conversation_id
            )
        else:  # Reject
            response = ACLMessage(
                performative=ACLMessage.Performative.REJECT_PROPOSAL,
                sender=self.agent_id,
                receiver=message.sender,
                content={"reason": "Better alternatives available"},
                conversation_id=message.conversation_id
            )
        
        logger.info(f"Agent {self.agent_id} responded to proposal")
    
    async def _handle_subscription(self, message: ACLMessage):
        """Handle subscription requests"""
        
        topic = message.content.get('topic')
        subscriber = message.sender
        
        self.subscriptions[topic].append(subscriber)
        logger.info(f"Agent {subscriber} subscribed to {topic}")
    
    async def _handle_inform(self, message: ACLMessage):
        """Handle information messages"""
        
        # Update knowledge base
        info = message.content
        topic = info.get('topic', 'general')
        self.knowledge_base[topic] = info
        
        # Learn from the information
        self.experience_buffer.append({
            'type': 'inform',
            'content': info,
            'source': message.sender,
            'timestamp': message.timestamp
        })
    
    def _extract_proposal_features(self, proposal: Dict[str, Any]) -> List[float]:
        """Extract features from proposal for decision making"""
        
        features = []
        features.append(proposal.get('price', 0))
        features.append(proposal.get('estimated_time', 0))
        features.append(proposal.get('confidence', 0))
        features.append(len(self.current_tasks))
        features.append(self.reputation)
        
        # Pad to 64 features
        features.extend([0] * (64 - len(features)))
        
        return features
    
    async def execute_swarm_behavior(self, behavior_name: str, neighbors: List['SwarmAgent']) -> Any:
        """Execute a specific swarm behavior"""
        
        if behavior_name in self.behaviors:
            return await self.behaviors[behavior_name].execute(self, neighbors)
        
        logger.warning(f"Unknown behavior: {behavior_name}")
        return None
    
    def update_position(self, new_position: np.ndarray):
        """Update agent position and track fitness"""
        
        self.position = new_position
        
        # Calculate fitness (task-dependent)
        fitness = self._calculate_fitness()
        
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.personal_best_position = self.position.copy()
    
    def _calculate_fitness(self) -> float:
        """Calculate agent fitness based on current state"""
        
        # Example fitness: completed tasks / total time
        if not self.task_history:
            return 0.0
        
        completed_tasks = sum(1 for t in self.task_history if t['status'] == 'completed')
        total_time = sum(t.get('duration', 1) for t in self.task_history)
        
        return completed_tasks / (total_time + 1)
    
    def get_available_moves(self) -> List[np.ndarray]:
        """Get available moves from current position"""
        
        # Generate possible moves in 3D space
        moves = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    move = self.position + np.array([dx, dy, dz])
                    moves.append(move)
        
        return moves
    
    def distance_to(self, position: np.ndarray) -> float:
        """Calculate distance to a position"""
        
        return np.linalg.norm(self.position - position)


class ContractNetProtocol:
    """Contract Net Protocol for task allocation"""
    
    def __init__(self):
        self.active_contracts = {}
        self.pending_cfps = {}
    
    async def initiate_contract(self, 
                               manager: SwarmAgent,
                               task: SwarmTask,
                               agents: List[SwarmAgent]) -> Optional[str]:
        """Initiate contract net protocol"""
        
        # Send CFP to all capable agents
        cfp = ACLMessage(
            performative=ACLMessage.Performative.CFP,
            sender=manager.agent_id,
            receiver="broadcast",
            content=task
        )
        
        responses = []
        
        # Collect proposals
        for agent in agents:
            if task.required_capabilities.issubset(agent.capabilities):
                await agent.receive_message(cfp)
                # In real implementation, wait for responses
                responses.append({
                    'agent_id': agent.agent_id,
                    'bid': random.random() * task.reward
                })
        
        if not responses:
            return None
        
        # Select best bid
        best_bid = min(responses, key=lambda x: x['bid'])
        
        # Award contract
        self.active_contracts[task.task_id] = {
            'task': task,
            'contractor': best_bid['agent_id'],
            'price': best_bid['bid'],
            'status': 'active'
        }
        
        return best_bid['agent_id']


class BlackboardSystem:
    """Blackboard architecture for knowledge sharing"""
    
    def __init__(self):
        self.blackboard = {
            'problems': {},
            'solutions': {},
            'knowledge': {},
            'constraints': {},
            'goals': {}
        }
        self.knowledge_sources = []
        self.controller = self._create_controller()
    
    def _create_controller(self):
        """Create blackboard controller"""
        
        class Controller:
            def __init__(self, blackboard):
                self.blackboard = blackboard
                self.activation_queue = asyncio.Queue()
            
            async def run(self):
                """Main control loop"""
                while True:
                    # Check which knowledge sources can contribute
                    for ks in self.blackboard.knowledge_sources:
                        if await ks.can_contribute(self.blackboard.blackboard):
                            await self.activation_queue.put(ks)
                    
                    # Activate knowledge sources
                    if not self.activation_queue.empty():
                        ks = await self.activation_queue.get()
                        await ks.contribute(self.blackboard.blackboard)
                    
                    await asyncio.sleep(0.1)
        
        return Controller(self)
    
    def register_knowledge_source(self, knowledge_source: Any):
        """Register a knowledge source"""
        self.knowledge_sources.append(knowledge_source)
    
    async def post_problem(self, problem_id: str, problem: Dict[str, Any]):
        """Post a problem to the blackboard"""
        self.blackboard['problems'][problem_id] = problem
    
    async def post_solution(self, problem_id: str, solution: Dict[str, Any]):
        """Post a solution to the blackboard"""
        self.blackboard['solutions'][problem_id] = solution
    
    def get_problems(self) -> Dict[str, Any]:
        """Get all problems"""
        return self.blackboard['problems']
    
    def get_solutions(self) -> Dict[str, Any]:
        """Get all solutions"""
        return self.blackboard['solutions']


class DistributedConsensus:
    """Raft consensus algorithm implementation"""
    
    class State(Enum):
        FOLLOWER = "follower"
        CANDIDATE = "candidate"
        LEADER = "leader"
    
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.state = self.State.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index = {peer: 0 for peer in peers}
        self.match_index = {peer: 0 for peer in peers}
        
        # Election timeout
        self.election_timeout = random.uniform(150, 300)
        self.last_heartbeat = datetime.now()
    
    async def start_election(self):
        """Start leader election"""
        
        self.state = self.State.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        
        votes = 1  # Vote for self
        
        # Request votes from peers
        for peer in self.peers:
            vote_request = {
                'term': self.current_term,
                'candidate_id': self.node_id,
                'last_log_index': len(self.log) - 1,
                'last_log_term': self.log[-1]['term'] if self.log else 0
            }
            
            # In real implementation, send RPC
            # For now, simulate random votes
            if random.random() > 0.3:
                votes += 1
        
        # Check if won election
        if votes > len(self.peers) / 2:
            self.state = self.State.LEADER
            await self.send_heartbeats()
        else:
            self.state = self.State.FOLLOWER
    
    async def send_heartbeats(self):
        """Send heartbeats as leader"""
        
        for peer in self.peers:
            heartbeat = {
                'term': self.current_term,
                'leader_id': self.node_id,
                'prev_log_index': self.next_index[peer] - 1,
                'prev_log_term': self.log[self.next_index[peer] - 1]['term'] if self.next_index[peer] > 0 else 0,
                'entries': [],
                'leader_commit': self.commit_index
            }
            
            # Send heartbeat (in real implementation)
            logger.debug(f"Leader {self.node_id} sending heartbeat to {peer}")
    
    async def append_entry(self, entry: Dict[str, Any]) -> bool:
        """Append entry to log (leader only)"""
        
        if self.state != self.State.LEADER:
            return False
        
        entry['term'] = self.current_term
        entry['index'] = len(self.log)
        self.log.append(entry)
        
        # Replicate to followers
        success_count = 1  # Self
        
        for peer in self.peers:
            # In real implementation, send AppendEntries RPC
            if random.random() > 0.2:  # Simulate success
                success_count += 1
        
        # Check if majority
        if success_count > len(self.peers) / 2:
            self.commit_index = len(self.log) - 1
            return True
        
        return False


class WorldClassSwarmSystem:
    """Main world-class swarm system orchestrator"""
    
    def __init__(self):
        self.agents: Dict[str, SwarmAgent] = {}
        self.contract_net = ContractNetProtocol()
        self.blackboard = BlackboardSystem()
        self.consensus = None
        self.swarm_graph = nx.Graph()
        
        # Publish-subscribe system
        self.pubsub = defaultdict(list)
        
        # Market mechanism
        self.task_market = []
        self.resource_prices = defaultdict(float)
        
        # Monitoring
        self.metrics = {
            'tasks_completed': 0,
            'average_completion_time': 0,
            'swarm_efficiency': 0,
            'consensus_rounds': 0
        }
    
    async def create_agent(self, 
                          agent_type: str,
                          capabilities: Set[str]) -> SwarmAgent:
        """Create a new swarm agent"""
        
        agent_id = f"{agent_type}_{len(self.agents)}"
        
        # Create agent with or without ray
        if RAY_AVAILABLE and ray is not None:
            # Create a Ray remote actor
            RemoteSwarmAgent = ray.remote(SwarmAgent)
            agent = RemoteSwarmAgent.remote(agent_id, agent_type, capabilities)
        else:
            agent = SwarmAgent(agent_id, agent_type, capabilities)
        
        self.agents[agent_id] = agent
        self.swarm_graph.add_node(agent_id)
        
        # Connect to nearby agents
        if len(self.agents) > 1:
            # Connect to 3-5 random existing agents
            num_connections = min(random.randint(3, 5), len(self.agents) - 1)
            neighbors = random.sample(
                [aid for aid in self.agents.keys() if aid != agent_id],
                num_connections
            )
            
            for neighbor in neighbors:
                self.swarm_graph.add_edge(agent_id, neighbor)
        
        logger.info(f"Created agent {agent_id} with capabilities {capabilities}")
        
        return agent
    
    async def submit_task(self, task: SwarmTask) -> str:
        """Submit task to swarm"""
        
        # Try contract net protocol first
        contractor = await self.contract_net.initiate_contract(
            list(self.agents.values())[0],  # Any agent can be manager
            task,
            list(self.agents.values())
        )
        
        if contractor:
            logger.info(f"Task {task.task_id} assigned to {contractor} via contract net")
            return contractor
        
        # Fallback to market mechanism
        self.task_market.append(task)
        
        # Trigger auction
        winner = await self._run_auction(task)
        
        return winner
    
    async def _run_auction(self, task: SwarmTask) -> str:
        """Run Dutch auction for task allocation"""
        
        starting_price = task.reward
        price_decrement = starting_price * 0.05
        current_price = starting_price
        
        while current_price > 0:
            # Check if any agent accepts current price
            for agent_id, agent in self.agents.items():
                if RAY_AVAILABLE:
                    agent_caps = await agent.capabilities.remote()
                else:
                    agent_caps = agent.capabilities
                
                if task.required_capabilities.issubset(agent_caps):
                    # Agent evaluates if price is acceptable
                    cost_estimate = task.estimated_complexity / len(agent_caps)
                    
                    if current_price >= cost_estimate:
                        logger.info(f"Agent {agent_id} won auction at price {current_price}")
                        return agent_id
            
            current_price -= price_decrement
        
        logger.warning(f"No agent accepted task {task.task_id}")
        return None
    
    async def execute_swarm_behavior(self, behavior: str):
        """Execute collective swarm behavior"""
        
        tasks = []
        
        for agent_id, agent in self.agents.items():
            # Get agent's neighbors
            neighbors = [
                self.agents[n] for n in self.swarm_graph.neighbors(agent_id)
            ]
            
            # Execute behavior
            if RAY_AVAILABLE:
                task = agent.execute_swarm_behavior.remote(behavior, neighbors)
                tasks.append(task)
            else:
                result = await agent.execute_swarm_behavior(behavior, neighbors)
                tasks.append(result)
        
        # Wait for all agents
        if RAY_AVAILABLE and ray is not None:
            results = await asyncio.gather(*[ray.get(t) for t in tasks])
        else:
            results = tasks
        
        logger.info(f"Executed {behavior} behavior across {len(self.agents)} agents")
        
        return results
    
    async def achieve_consensus(self, proposal: Dict[str, Any]) -> bool:
        """Achieve distributed consensus on proposal"""
        
        if not self.consensus:
            node_ids = list(self.agents.keys())
            self.consensus = DistributedConsensus(node_ids[0], node_ids[1:])
        
        # Leader election if needed
        if self.consensus.state == DistributedConsensus.State.FOLLOWER:
            await self.consensus.start_election()
        
        # Append to distributed log
        if self.consensus.state == DistributedConsensus.State.LEADER:
            success = await self.consensus.append_entry(proposal)
            
            self.metrics['consensus_rounds'] += 1
            
            return success
        
        return False
    
    async def publish(self, topic: str, message: Any):
        """Publish message to topic"""
        
        subscribers = self.pubsub[topic]
        
        for subscriber in subscribers:
            if subscriber in self.agents:
                agent = self.agents[subscriber]
                
                inform_msg = ACLMessage(
                    performative=ACLMessage.Performative.INFORM,
                    sender="system",
                    receiver=subscriber,
                    content={'topic': topic, 'message': message}
                )
                
                if RAY_AVAILABLE:
                    await agent.receive_message.remote(inform_msg)
                else:
                    await agent.receive_message(inform_msg)
    
    async def subscribe(self, agent_id: str, topic: str):
        """Subscribe agent to topic"""
        
        self.pubsub[topic].append(agent_id)
        logger.info(f"Agent {agent_id} subscribed to {topic}")
    
    def get_swarm_metrics(self) -> Dict[str, Any]:
        """Get swarm performance metrics"""
        
        metrics = self.metrics.copy()
        
        # Calculate additional metrics
        metrics['num_agents'] = len(self.agents)
        metrics['avg_connections'] = np.mean([
            self.swarm_graph.degree(n) for n in self.swarm_graph.nodes()
        ]) if self.swarm_graph.nodes() else 0
        
        metrics['clustering_coefficient'] = nx.average_clustering(self.swarm_graph)
        
        return metrics


# Example usage and testing
async def test_world_class_swarm():
    """Test the world-class swarm system"""
    
    # Initialize Ray if available
    if RAY_AVAILABLE and ray is not None:
        ray.init(ignore_reinit_error=True)
    
    # Create swarm system
    swarm = WorldClassSwarmSystem()
    
    # Create diverse agents
    agent_types = [
        ("analyzer", {"analysis", "reasoning", "planning"}),
        ("executor", {"execution", "coding", "debugging"}),
        ("coordinator", {"coordination", "communication", "negotiation"}),
        ("optimizer", {"optimization", "learning", "adaptation"}),
        ("monitor", {"monitoring", "logging", "reporting"})
    ]
    
    for agent_type, capabilities in agent_types * 3:  # Create 3 of each type
        await swarm.create_agent(agent_type, capabilities)
    
    # Test contract net protocol
    task1 = SwarmTask(
        task_id="task_001",
        task_type="analysis",
        payload={"analyze": "system performance"},
        required_capabilities={"analysis", "reasoning"},
        estimated_complexity=2.0,
        reward=10.0
    )
    
    contractor = await swarm.submit_task(task1)
    logger.info(f"Task assigned to: {contractor}")
    
    # Test swarm behaviors
    await swarm.execute_swarm_behavior("flocking")
    await swarm.execute_swarm_behavior("pso")
    
    # Test consensus
    proposal = {"action": "update_parameters", "values": {"learning_rate": 0.01}}
    consensus_achieved = await swarm.achieve_consensus(proposal)
    logger.info(f"Consensus achieved: {consensus_achieved}")
    
    # Test publish-subscribe
    await swarm.subscribe("analyzer_0", "performance_updates")
    await swarm.publish("performance_updates", {"cpu": 45, "memory": 60})
    
    # Get metrics
    metrics = swarm.get_swarm_metrics()
    logger.info(f"Swarm metrics: {metrics}")
    
    if RAY_AVAILABLE and ray is not None:
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(test_world_class_swarm())