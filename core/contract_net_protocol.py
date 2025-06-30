"""
World-Class Contract Net Protocol Implementation
===============================================

An advanced implementation of the Contract Net Protocol for multi-agent systems with:
- Dynamic task decomposition and allocation
- Multi-attribute negotiation and bidding
- Reputation-based contractor selection
- Contract monitoring and enforcement
- Adaptive learning from contract outcomes
- Support for complex task dependencies
- Distributed consensus mechanisms
- Fault tolerance and recovery
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
import numpy as np
from collections import defaultdict, deque
import networkx as nx
from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge, Summary
import hashlib

logger = get_logger(__name__)

# Metrics
contracts_initiated = Counter("contracts_initiated_total", "Total contracts initiated")
contracts_completed = Counter(
    "contracts_completed_total", "Total contracts completed", ["status"]
)
bidding_rounds = Histogram("bidding_rounds_duration_seconds", "Bidding round duration")
contract_values = Summary("contract_values", "Contract values distribution")
task_completion_time = Histogram(
    "task_completion_duration_seconds", "Task completion time"
)
agent_reputation_scores = Gauge(
    "agent_reputation_scores", "Agent reputation scores", ["agent_id"]
)


class ContractStatus(Enum):
    """Contract lifecycle states"""

    DRAFT = "draft"
    ANNOUNCED = "announced"
    BIDDING = "bidding"
    EVALUATING = "evaluating"
    AWARDED = "awarded"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DISPUTED = "disputed"


class BidStrategy(Enum):
    """Bidding strategies"""

    COMPETITIVE = "competitive"  # Lowest cost
    COLLABORATIVE = "collaborative"  # Best for team
    QUALITY_FOCUSED = "quality_focused"  # Highest quality
    TIME_OPTIMIZED = "time_optimized"  # Fastest completion
    BALANCED = "balanced"  # Balanced approach


@dataclass
class TaskSpecification:
    """Detailed task specification for contracts"""

    task_id: str
    task_type: str
    description: str
    requirements: List[str]
    constraints: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: float = 1.0
    deadline: Optional[datetime] = None
    budget: Optional[float] = None
    quality_threshold: float = 0.8
    required_capabilities: Set[str] = field(default_factory=set)
    preferred_capabilities: Set[str] = field(default_factory=set)
    deliverables: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_criteria: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractTerms:
    """Contract terms and conditions"""

    payment: float
    deadline: datetime
    quality_requirements: Dict[str, float]
    penalties: Dict[str, float] = field(default_factory=dict)
    incentives: Dict[str, float] = field(default_factory=dict)
    termination_conditions: List[str] = field(default_factory=list)
    dispute_resolution: str = "consensus"
    payment_schedule: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Bid:
    """Agent bid for a contract"""

    bid_id: str
    agent_id: str
    task_id: str
    proposed_cost: float
    estimated_completion_time: timedelta
    quality_guarantee: float
    capabilities_match: float
    technical_approach: str
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    terms_acceptance: Dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Contract:
    """Complete contract information"""

    contract_id: str
    task_specification: TaskSpecification
    manager_id: str
    contractor_id: Optional[str] = None
    terms: Optional[ContractTerms] = None
    status: ContractStatus = ContractStatus.DRAFT
    bids: List[Bid] = field(default_factory=list)
    negotiation_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    awarded_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    subcontracts: List[str] = field(default_factory=list)


@dataclass
class AgentProfile:
    """Agent profile for contract negotiations"""

    agent_id: str
    capabilities: Set[str]
    reputation_score: float = 1.0
    completed_contracts: int = 0
    success_rate: float = 1.0
    average_quality_score: float = 0.8
    specializations: Dict[str, float] = field(default_factory=dict)
    availability: float = 1.0
    preferred_contract_types: Set[str] = field(default_factory=set)
    historical_performance: List[Dict[str, Any]] = field(default_factory=list)
    current_workload: int = 0
    max_concurrent_contracts: int = 5


class ContractEvaluator:
    """Evaluates bids and contractor performance"""

    def __init__(
        self,
        reputation_weight: float = 0.3,
        cost_weight: float = 0.3,
        quality_weight: float = 0.2,
        time_weight: float = 0.2,
    ):

        self.weights = {
            "reputation": reputation_weight,
            "cost": cost_weight,
            "quality": quality_weight,
            "time": time_weight,
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    async def evaluate_bid(
        self,
        bid: Bid,
        task: TaskSpecification,
        agent_profile: AgentProfile,
        other_bids: List[Bid],
    ) -> float:
        """Evaluate a bid comprehensively"""
        scores = {}

        # Reputation score
        scores["reputation"] = agent_profile.reputation_score

        # Cost score (normalized against other bids)
        if other_bids:
            min_cost = min(b.proposed_cost for b in other_bids)
            max_cost = max(b.proposed_cost for b in other_bids)
            if max_cost > min_cost:
                scores["cost"] = 1 - (bid.proposed_cost - min_cost) / (
                    max_cost - min_cost
                )
            else:
                scores["cost"] = 1.0
        else:
            scores["cost"] = (
                1.0 if bid.proposed_cost <= (task.budget or float("inf")) else 0.5
            )

        # Quality score
        scores["quality"] = bid.quality_guarantee * agent_profile.average_quality_score

        # Time score
        if task.deadline:
            time_available = (task.deadline - datetime.now()).total_seconds()
            time_needed = bid.estimated_completion_time.total_seconds()
            scores["time"] = min(1.0, time_available / max(time_needed, 1))
        else:
            scores["time"] = 0.8  # Default score when no deadline

        # Calculate weighted score
        total_score = sum(scores[k] * self.weights[k] for k in self.weights)

        # Apply bonuses/penalties

        # Capability match bonus
        capability_bonus = bid.capabilities_match * 0.1
        total_score += capability_bonus

        # Workload penalty
        workload_penalty = (
            agent_profile.current_workload / agent_profile.max_concurrent_contracts
        ) * 0.1
        total_score -= workload_penalty

        # Specialization bonus
        if task.task_type in agent_profile.specializations:
            specialization_bonus = agent_profile.specializations[task.task_type] * 0.05
            total_score += specialization_bonus

        return min(1.0, max(0.0, total_score))

    async def rank_bids(
        self,
        bids: List[Bid],
        task: TaskSpecification,
        agent_profiles: Dict[str, AgentProfile],
    ) -> List[Tuple[Bid, float]]:
        """Rank all bids for a task"""
        scored_bids = []

        for bid in bids:
            profile = agent_profiles.get(bid.agent_id)
            if profile:
                score = await self.evaluate_bid(bid, task, profile, bids)
                scored_bids.append((bid, score))

        # Sort by score (highest first)
        scored_bids.sort(key=lambda x: x[1], reverse=True)

        return scored_bids


class NegotiationProtocol:
    """Handles multi-round negotiations"""

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.negotiation_history = defaultdict(list)

    async def negotiate(
        self,
        contract: Contract,
        initial_bid: Bid,
        manager_agent: Any,
        contractor_agent: Any,
    ) -> Tuple[bool, ContractTerms]:
        """Conduct negotiation between manager and contractor"""
        current_terms = contract.terms or self._create_initial_terms(
            contract, initial_bid
        )

        for round_num in range(self.max_rounds):
            # Manager proposes terms
            manager_proposal = await manager_agent.propose_terms(
                contract, current_terms, initial_bid
            )

            # Contractor responds
            contractor_response = await contractor_agent.evaluate_terms(
                contract, manager_proposal
            )

            negotiation_round = {
                "round": round_num + 1,
                "manager_proposal": manager_proposal,
                "contractor_response": contractor_response,
                "timestamp": datetime.now(),
            }

            self.negotiation_history[contract.contract_id].append(negotiation_round)
            contract.negotiation_history.append(negotiation_round)

            if contractor_response["accepted"]:
                return True, manager_proposal

            # Contractor counter-proposes
            if "counter_proposal" in contractor_response:
                current_terms = contractor_response["counter_proposal"]
            else:
                # No counter-proposal means negotiation failed
                return False, current_terms

        # Max rounds reached without agreement
        return False, current_terms

    def _create_initial_terms(self, contract: Contract, bid: Bid) -> ContractTerms:
        """Create initial contract terms from bid"""
        return ContractTerms(
            payment=bid.proposed_cost,
            deadline=datetime.now() + bid.estimated_completion_time,
            quality_requirements={"overall": bid.quality_guarantee},
            payment_schedule=[
                {"milestone": "start", "percentage": 25},
                {"milestone": "midpoint", "percentage": 25},
                {"milestone": "completion", "percentage": 50},
            ],
        )


class ContractMonitor:
    """Monitors contract execution and performance"""

    def __init__(self):
        self.monitoring_data = defaultdict(list)
        self.alerts = defaultdict(list)
        self.performance_thresholds = {
            "progress_delay": 0.2,  # 20% behind schedule
            "quality_drop": 0.1,  # 10% below guaranteed quality
            "cost_overrun": 0.15,  # 15% over budget
        }

    async def monitor_contract(self, contract: Contract, current_state: Dict[str, Any]):
        """Monitor contract execution"""
        monitoring_entry = {
            "contract_id": contract.contract_id,
            "timestamp": datetime.now(),
            "state": current_state,
            "alerts": [],
        }

        # Check progress
        if "progress" in current_state:
            expected_progress = self._calculate_expected_progress(contract)
            if (
                current_state["progress"]
                < expected_progress - self.performance_thresholds["progress_delay"]
            ):
                alert = {
                    "type": "progress_delay",
                    "severity": "warning",
                    "message": f"Progress behind schedule: {current_state['progress']:.1%} vs {expected_progress:.1%} expected",
                }
                monitoring_entry["alerts"].append(alert)
                self.alerts[contract.contract_id].append(alert)

        # Check quality
        if "quality_metrics" in current_state:
            for metric, value in current_state["quality_metrics"].items():
                required = contract.terms.quality_requirements.get(metric, 0)
                if value < required - self.performance_thresholds["quality_drop"]:
                    alert = {
                        "type": "quality_issue",
                        "severity": "critical",
                        "message": f"Quality below requirement: {metric}={value:.2f} vs {required:.2f} required",
                    }
                    monitoring_entry["alerts"].append(alert)
                    self.alerts[contract.contract_id].append(alert)

        # Check cost
        if "current_cost" in current_state:
            if current_state["current_cost"] > contract.terms.payment * (
                1 + self.performance_thresholds["cost_overrun"]
            ):
                alert = {
                    "type": "cost_overrun",
                    "severity": "warning",
                    "message": f"Cost overrun: ${current_state['current_cost']:.2f} vs ${contract.terms.payment:.2f} budgeted",
                }
                monitoring_entry["alerts"].append(alert)
                self.alerts[contract.contract_id].append(alert)

        self.monitoring_data[contract.contract_id].append(monitoring_entry)

        # Update contract performance metrics
        contract.performance_metrics.update(
            {
                "current_progress": current_state.get("progress", 0),
                "current_quality": (
                    np.mean(list(current_state.get("quality_metrics", {}).values()))
                    if current_state.get("quality_metrics")
                    else 0
                ),
                "current_cost": current_state.get("current_cost", 0),
                "alert_count": len(self.alerts[contract.contract_id]),
            }
        )

    def _calculate_expected_progress(self, contract: Contract) -> float:
        """Calculate expected progress based on time elapsed"""
        if not contract.awarded_at or not contract.terms:
            return 0.0

        total_duration = (contract.terms.deadline - contract.awarded_at).total_seconds()
        elapsed = (datetime.now() - contract.awarded_at).total_seconds()

        return min(1.0, elapsed / max(total_duration, 1))

    async def get_contract_health(self, contract_id: str) -> Dict[str, Any]:
        """Get overall contract health status"""
        alerts = self.alerts.get(contract_id, [])
        recent_alerts = [
            a
            for a in alerts
            if (datetime.now() - a.get("timestamp", datetime.now()))
            < timedelta(hours=24)
        ]

        critical_count = sum(
            1 for a in recent_alerts if a.get("severity") == "critical"
        )
        warning_count = sum(1 for a in recent_alerts if a.get("severity") == "warning")

        if critical_count > 0:
            health_status = "critical"
        elif warning_count > 2:
            health_status = "warning"
        else:
            health_status = "healthy"

        return {
            "status": health_status,
            "critical_alerts": critical_count,
            "warning_alerts": warning_count,
            "recent_alerts": recent_alerts[-5:],  # Last 5 alerts
        }


class ReputationManager:
    """Manages agent reputation based on contract performance"""

    def __init__(self, initial_reputation: float = 1.0, learning_rate: float = 0.1):

        self.reputations = defaultdict(lambda: initial_reputation)
        self.learning_rate = learning_rate
        self.performance_history = defaultdict(list)

    async def update_reputation(
        self, agent_id: str, contract: Contract, outcome: Dict[str, Any]
    ):
        """Update agent reputation based on contract outcome"""
        performance_score = self._calculate_performance_score(contract, outcome)

        # Store performance history
        self.performance_history[agent_id].append(
            {
                "contract_id": contract.contract_id,
                "score": performance_score,
                "timestamp": datetime.now(),
                "contract_type": contract.task_specification.task_type,
            }
        )

        # Update reputation using exponential moving average
        current_reputation = self.reputations[agent_id]
        new_reputation = (
            1 - self.learning_rate
        ) * current_reputation + self.learning_rate * performance_score

        # Apply bounds
        new_reputation = max(0.0, min(1.0, new_reputation))

        self.reputations[agent_id] = new_reputation

        # Update metric
        agent_reputation_scores.labels(agent_id=agent_id).set(new_reputation)

        logger.info(
            "Updated agent reputation",
            agent_id=agent_id,
            old_reputation=current_reputation,
            new_reputation=new_reputation,
            performance_score=performance_score,
        )

    def _calculate_performance_score(
        self, contract: Contract, outcome: Dict[str, Any]
    ) -> float:
        """Calculate performance score from contract outcome"""
        scores = []

        # Time performance
        if outcome.get("completed_on_time", False):
            scores.append(1.0)
        else:
            delay_penalty = outcome.get("delay_percentage", 0) / 100
            scores.append(max(0, 1 - delay_penalty))

        # Quality performance
        quality_score = outcome.get("quality_score", 0.8)
        required_quality = (
            contract.terms.quality_requirements.get("overall", 0.8)
            if contract.terms
            else 0.8
        )
        scores.append(min(1.0, quality_score / required_quality))

        # Cost performance
        if outcome.get("within_budget", True):
            scores.append(1.0)
        else:
            overrun_penalty = outcome.get("cost_overrun_percentage", 0) / 100
            scores.append(max(0, 1 - overrun_penalty))

        # Contract completion
        if outcome.get("status") == "completed":
            scores.append(1.0)
        elif outcome.get("status") == "partial":
            scores.append(0.5)
        else:
            scores.append(0.0)

        return np.mean(scores)

    def get_agent_reputation(self, agent_id: str) -> float:
        """Get current agent reputation"""
        return self.reputations[agent_id]

    def get_specialized_reputation(self, agent_id: str, contract_type: str) -> float:
        """Get reputation for specific contract type"""
        history = self.performance_history[agent_id]
        relevant_scores = [
            h["score"] for h in history if h.get("contract_type") == contract_type
        ]

        if not relevant_scores:
            return self.reputations[agent_id]

        # Weight recent performance more heavily
        weights = np.exp(np.linspace(-1, 0, len(relevant_scores)))
        weights /= weights.sum()

        return np.average(relevant_scores, weights=weights)


class ContractNetProtocol:
    """
    World-class Contract Net Protocol implementation
    """

    def __init__(
        self,
        enable_negotiation: bool = True,
        enable_monitoring: bool = True,
        enable_subcontracting: bool = True,
        bid_timeout: timedelta = timedelta(minutes=5),
    ):

        self.enable_negotiation = enable_negotiation
        self.enable_monitoring = enable_monitoring
        self.enable_subcontracting = enable_subcontracting
        self.bid_timeout = bid_timeout

        # Core components
        self.contracts: Dict[str, Contract] = {}
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.evaluator = ContractEvaluator()
        self.negotiator = NegotiationProtocol()
        self.monitor = ContractMonitor()
        self.reputation_manager = ReputationManager()

        # Task dependency graph
        self.task_graph = nx.DiGraph()

        # Contract queues
        self.pending_contracts = asyncio.Queue()
        self.active_contracts: Dict[str, Contract] = {}
        self.completed_contracts: Dict[str, Contract] = {}

        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        logger.info(
            "Contract Net Protocol initialized",
            negotiation=enable_negotiation,
            monitoring=enable_monitoring,
            subcontracting=enable_subcontracting,
        )

    def register_agent(self, agent_profile: AgentProfile):
        """Register an agent in the system"""
        self.agent_profiles[agent_profile.agent_id] = agent_profile
        logger.info(
            "Agent registered",
            agent_id=agent_profile.agent_id,
            capabilities=list(agent_profile.capabilities),
        )

    async def initiate_contract(
        self, task_spec: TaskSpecification, manager_id: str
    ) -> Contract:
        """
        Initiate a new contract for a task

        Args:
            task_spec: Task specification
            manager_id: ID of the manager agent

        Returns:
            Created contract
        """
        contract_id = f"contract_{uuid.uuid4().hex[:8]}"

        contract = Contract(
            contract_id=contract_id,
            task_specification=task_spec,
            manager_id=manager_id,
            status=ContractStatus.DRAFT,
        )

        self.contracts[contract_id] = contract

        # Add to task graph
        self.task_graph.add_node(task_spec.task_id, contract=contract)

        # Add dependencies
        for dep in task_spec.dependencies:
            if dep in self.task_graph:
                self.task_graph.add_edge(dep, task_spec.task_id)

        # Update metrics
        contracts_initiated.inc()

        logger.info(
            "Contract initiated",
            contract_id=contract_id,
            task_type=task_spec.task_type,
            manager_id=manager_id,
        )

        return contract

    async def announce_contract(self, contract: Contract) -> List[str]:
        """
        Announce contract to eligible agents

        Returns:
            List of agent IDs that received the announcement
        """
        contract.status = ContractStatus.ANNOUNCED
        eligible_agents = []

        # Find eligible agents
        for agent_id, profile in self.agent_profiles.items():
            if agent_id == contract.manager_id:
                continue

            # Check capabilities
            if contract.task_specification.required_capabilities.issubset(
                profile.capabilities
            ):
                # Check availability
                if profile.current_workload < profile.max_concurrent_contracts:
                    eligible_agents.append(agent_id)

        # Trigger announcement event
        await self._trigger_event("contract_announced", contract, eligible_agents)

        logger.info(
            "Contract announced",
            contract_id=contract.contract_id,
            eligible_agents=len(eligible_agents),
        )

        return eligible_agents

    async def submit_bid(self, bid: Bid, contract_id: str) -> bool:
        """
        Submit a bid for a contract

        Returns:
            Whether bid was accepted
        """
        contract = self.contracts.get(contract_id)
        if not contract:
            return False

        if contract.status not in [ContractStatus.ANNOUNCED, ContractStatus.BIDDING]:
            return False

        # Validate bid
        if not await self._validate_bid(bid, contract):
            return False

        contract.status = ContractStatus.BIDDING
        contract.bids.append(bid)

        logger.info(
            "Bid submitted",
            contract_id=contract_id,
            agent_id=bid.agent_id,
            proposed_cost=bid.proposed_cost,
        )

        return True

    async def evaluate_bids(self, contract_id: str) -> Optional[str]:
        """
        Evaluate all bids and select winner

        Returns:
            Winning agent ID or None
        """
        contract = self.contracts.get(contract_id)
        if not contract or not contract.bids:
            return None

        contract.status = ContractStatus.EVALUATING

        with bidding_rounds.time():
            # Rank bids
            ranked_bids = await self.evaluator.rank_bids(
                contract.bids, contract.task_specification, self.agent_profiles
            )

            if not ranked_bids:
                contract.status = ContractStatus.FAILED
                return None

            # Select winner (may involve negotiation)
            winner = None

            for bid, score in ranked_bids:
                if self.enable_negotiation:
                    # Attempt negotiation
                    manager = await self._get_agent(contract.manager_id)
                    contractor = await self._get_agent(bid.agent_id)

                    if manager and contractor:
                        success, terms = await self.negotiator.negotiate(
                            contract, bid, manager, contractor
                        )

                        if success:
                            contract.contractor_id = bid.agent_id
                            contract.terms = terms
                            winner = bid.agent_id
                            break
                else:
                    # Direct selection without negotiation
                    contract.contractor_id = bid.agent_id
                    contract.terms = self._create_terms_from_bid(bid, contract)
                    winner = bid.agent_id
                    break

            if winner:
                contract.status = ContractStatus.AWARDED
                contract.awarded_at = datetime.now()

                # Update agent workload
                if winner in self.agent_profiles:
                    self.agent_profiles[winner].current_workload += 1

                # Update metrics
                contract_values.observe(contract.terms.payment)

                # Trigger event
                await self._trigger_event("contract_awarded", contract, winner)

                logger.info(
                    "Contract awarded",
                    contract_id=contract_id,
                    winner=winner,
                    value=contract.terms.payment,
                )
            else:
                contract.status = ContractStatus.FAILED
                contracts_completed.labels(status="failed").inc()

            return winner

    async def start_contract_execution(self, contract_id: str) -> bool:
        """
        Start contract execution
        """
        contract = self.contracts.get(contract_id)
        if not contract or contract.status != ContractStatus.AWARDED:
            return False

        contract.status = ContractStatus.IN_PROGRESS
        self.active_contracts[contract_id] = contract

        # Start monitoring if enabled
        if self.enable_monitoring:
            asyncio.create_task(self._monitor_contract(contract))

        # Check if subcontracting is needed
        if self.enable_subcontracting:
            await self._check_subcontracting_needs(contract)

        await self._trigger_event("contract_started", contract)

        return True

    async def report_contract_progress(
        self, contract_id: str, progress_report: Dict[str, Any]
    ):
        """Report progress on contract execution"""
        contract = self.active_contracts.get(contract_id)
        if not contract:
            return

        # Monitor progress
        if self.enable_monitoring:
            await self.monitor.monitor_contract(contract, progress_report)

        # Check if contract is complete
        if progress_report.get("status") == "completed":
            await self.complete_contract(contract_id, progress_report)

        await self._trigger_event("contract_progress", contract, progress_report)

    async def complete_contract(self, contract_id: str, outcome: Dict[str, Any]):
        """Complete a contract"""
        contract = self.active_contracts.get(contract_id)
        if not contract:
            return

        contract.status = ContractStatus.COMPLETED
        contract.completed_at = datetime.now()

        # Move to completed
        self.completed_contracts[contract_id] = contract
        del self.active_contracts[contract_id]

        # Update agent workload
        if contract.contractor_id in self.agent_profiles:
            self.agent_profiles[contract.contractor_id].current_workload -= 1

        # Update reputation
        await self.reputation_manager.update_reputation(
            contract.contractor_id, contract, outcome
        )

        # Update metrics
        contracts_completed.labels(status="completed").inc()
        if contract.awarded_at:
            completion_time = (
                contract.completed_at - contract.awarded_at
            ).total_seconds()
            task_completion_time.observe(completion_time)

        await self._trigger_event("contract_completed", contract, outcome)

        logger.info(
            "Contract completed",
            contract_id=contract_id,
            contractor=contract.contractor_id,
            quality_score=outcome.get("quality_score"),
        )

    async def handle_contract_failure(self, contract_id: str, reason: str):
        """Handle contract failure"""
        contract = self.active_contracts.get(contract_id)
        if not contract:
            return

        contract.status = ContractStatus.FAILED

        # Update reputation negatively
        if contract.contractor_id:
            await self.reputation_manager.update_reputation(
                contract.contractor_id, contract, {"status": "failed", "reason": reason}
            )

        # Re-announce if possible
        if (
            contract.task_specification.deadline
            and contract.task_specification.deadline > datetime.now()
        ):
            # Create new contract for the same task
            new_contract = await self.initiate_contract(
                contract.task_specification, contract.manager_id
            )
            await self.announce_contract(new_contract)

        contracts_completed.labels(status="failed").inc()

        await self._trigger_event("contract_failed", contract, reason)

    async def get_contract_status(self, contract_id: str) -> Dict[str, Any]:
        """Get comprehensive contract status"""
        contract = self.contracts.get(contract_id)
        if not contract:
            return {"error": "Contract not found"}

        status = {
            "contract_id": contract_id,
            "status": contract.status.value,
            "task_type": contract.task_specification.task_type,
            "manager": contract.manager_id,
            "contractor": contract.contractor_id,
            "created_at": contract.created_at.isoformat(),
            "bid_count": len(contract.bids),
            "performance_metrics": contract.performance_metrics,
        }

        if contract.awarded_at:
            status["awarded_at"] = contract.awarded_at.isoformat()

        if contract.completed_at:
            status["completed_at"] = contract.completed_at.isoformat()

        if self.enable_monitoring and contract_id in self.monitor.alerts:
            status["health"] = await self.monitor.get_contract_health(contract_id)

        return status

    def subscribe_to_event(self, event_type: str, handler: Callable):
        """Subscribe to contract events"""
        self.event_handlers[event_type].append(handler)

    # Private methods

    async def _validate_bid(self, bid: Bid, contract: Contract) -> bool:
        """Validate bid against contract requirements"""
        # Check if agent exists
        if bid.agent_id not in self.agent_profiles:
            return False

        profile = self.agent_profiles[bid.agent_id]

        # Check capabilities
        if not contract.task_specification.required_capabilities.issubset(
            profile.capabilities
        ):
            return False

        # Check budget constraints
        if (
            contract.task_specification.budget
            and bid.proposed_cost > contract.task_specification.budget
        ):
            return False

        # Check deadline feasibility
        if contract.task_specification.deadline:
            estimated_completion = datetime.now() + bid.estimated_completion_time
            if estimated_completion > contract.task_specification.deadline:
                return False

        return True

    def _create_terms_from_bid(self, bid: Bid, contract: Contract) -> ContractTerms:
        """Create contract terms from winning bid"""
        return ContractTerms(
            payment=bid.proposed_cost,
            deadline=datetime.now() + bid.estimated_completion_time,
            quality_requirements={
                "overall": bid.quality_guarantee,
                **contract.task_specification.evaluation_criteria,
            },
        )

    async def _get_agent(self, agent_id: str) -> Optional[Any]:
        """Get agent instance (placeholder for actual implementation)"""
        # In real implementation, this would retrieve the actual agent
        return None

    async def _monitor_contract(self, contract: Contract):
        """Monitor contract execution"""
        while contract.status == ContractStatus.IN_PROGRESS:
            try:
                # Wait for monitoring interval
                await asyncio.sleep(60)  # Check every minute

                # Get current state (would be reported by contractor)
                # This is a placeholder - actual implementation would query the contractor
                current_state = {
                    "progress": 0.5,  # Placeholder
                    "quality_metrics": {"overall": 0.85},
                    "current_cost": contract.terms.payment * 0.6,
                }

                await self.monitor.monitor_contract(contract, current_state)

                # Check for critical alerts
                health = await self.monitor.get_contract_health(contract.contract_id)
                if health["status"] == "critical":
                    await self._trigger_event("contract_alert", contract, health)

            except Exception as e:
                logger.error(
                    "Contract monitoring error",
                    contract_id=contract.contract_id,
                    error=str(e),
                )

    async def _check_subcontracting_needs(self, contract: Contract):
        """Check if subcontracting is needed"""
        # Analyze task complexity
        task_components = await self._decompose_task(contract.task_specification)

        if len(task_components) > 1:
            # Create subcontracts
            for component in task_components:
                subcontract = await self.initiate_contract(
                    component,
                    contract.contractor_id,  # Contractor becomes manager of subcontract
                )
                contract.subcontracts.append(subcontract.contract_id)

                # Announce subcontract
                await self.announce_contract(subcontract)

    async def _decompose_task(
        self, task_spec: TaskSpecification
    ) -> List[TaskSpecification]:
        """Decompose task into subtasks if needed"""
        # Placeholder - actual implementation would use AI or rules
        return []

    async def _trigger_event(self, event_type: str, *args, **kwargs):
        """Trigger event handlers"""
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error("Event handler error", event_type=event_type, error=str(e))


# Example usage
async def example_usage():
    """Example usage of the Contract Net Protocol"""

    # Initialize protocol
    cnp = ContractNetProtocol(
        enable_negotiation=True, enable_monitoring=True, enable_subcontracting=True
    )

    # Register agents
    agents = [
        AgentProfile(
            agent_id="agent_1",
            capabilities={"data_processing", "analysis", "reporting"},
            reputation_score=0.95,
            specializations={"data_processing": 0.9},
        ),
        AgentProfile(
            agent_id="agent_2",
            capabilities={"data_processing", "visualization"},
            reputation_score=0.85,
            specializations={"visualization": 0.95},
        ),
        AgentProfile(
            agent_id="agent_3",
            capabilities={"analysis", "ml_modeling", "reporting"},
            reputation_score=0.90,
            specializations={"ml_modeling": 0.85},
        ),
    ]

    for agent in agents:
        cnp.register_agent(agent)

    # Create task specification
    task_spec = TaskSpecification(
        task_id="task_001",
        task_type="data_processing",
        description="Process and analyze customer data",
        requirements=[
            "Clean and validate data",
            "Generate statistical analysis",
            "Create visualization report",
        ],
        constraints={"max_processing_time": "2 hours"},
        priority=0.8,
        deadline=datetime.now() + timedelta(hours=24),
        budget=1000.0,
        required_capabilities={"data_processing", "analysis"},
        preferred_capabilities={"visualization"},
        deliverables=[
            {"type": "cleaned_data", "format": "csv"},
            {"type": "analysis_report", "format": "pdf"},
            {"type": "visualizations", "format": "html"},
        ],
        evaluation_criteria={
            "accuracy": 0.95,
            "completeness": 0.98,
            "presentation": 0.90,
        },
    )

    # Manager initiates contract
    contract = await cnp.initiate_contract(task_spec, "manager_agent")

    # Announce to agents
    eligible_agents = await cnp.announce_contract(contract)
    print(f"Contract announced to {len(eligible_agents)} agents")

    # Agents submit bids
    bids = [
        Bid(
            bid_id="bid_1",
            agent_id="agent_1",
            task_id=task_spec.task_id,
            proposed_cost=800.0,
            estimated_completion_time=timedelta(hours=20),
            quality_guarantee=0.92,
            capabilities_match=0.9,
            technical_approach="Automated pipeline with quality checks",
            confidence=0.85,
        ),
        Bid(
            bid_id="bid_2",
            agent_id="agent_2",
            task_id=task_spec.task_id,
            proposed_cost=950.0,
            estimated_completion_time=timedelta(hours=18),
            quality_guarantee=0.88,
            capabilities_match=0.8,
            technical_approach="Semi-automated with manual validation",
            confidence=0.75,
        ),
        Bid(
            bid_id="bid_3",
            agent_id="agent_3",
            task_id=task_spec.task_id,
            proposed_cost=850.0,
            estimated_completion_time=timedelta(hours=22),
            quality_guarantee=0.95,
            capabilities_match=0.85,
            technical_approach="ML-enhanced processing pipeline",
            confidence=0.90,
        ),
    ]

    for bid in bids:
        await cnp.submit_bid(bid, contract.contract_id)

    # Evaluate bids and award contract
    winner = await cnp.evaluate_bids(contract.contract_id)
    print(f"Contract awarded to: {winner}")

    # Start execution
    await cnp.start_contract_execution(contract.contract_id)

    # Simulate progress reports
    await cnp.report_contract_progress(
        contract.contract_id,
        {
            "progress": 0.25,
            "quality_metrics": {"accuracy": 0.94, "completeness": 0.96},
            "current_cost": 200.0,
        },
    )

    # Get status
    status = await cnp.get_contract_status(contract.contract_id)
    print(f"Contract status: {json.dumps(status, indent=2)}")

    # Complete contract
    await cnp.complete_contract(
        contract.contract_id,
        {
            "status": "completed",
            "completed_on_time": True,
            "quality_score": 0.93,
            "within_budget": True,
            "deliverables": ["cleaned_data.csv", "analysis_report.pdf", "visuals.html"],
        },
    )

    # Check updated reputation
    for agent_id in ["agent_1", "agent_2", "agent_3"]:
        reputation = cnp.reputation_manager.get_agent_reputation(agent_id)
        print(f"{agent_id} reputation: {reputation:.3f}")


if __name__ == "__main__":
    asyncio.run(example_usage())
