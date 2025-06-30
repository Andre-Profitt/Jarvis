"""
Knowledge Base Tool for JARVIS
==============================

Provides advanced knowledge management capabilities including storage,
retrieval, reasoning, and knowledge graph operations.
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import aiofiles
from collections import defaultdict
import re
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseTool, ToolMetadata, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge entries"""

    FACT = "fact"
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    RULE = "rule"
    EXPERIENCE = "experience"
    REFERENCE = "reference"
    RELATIONSHIP = "relationship"


class ReasoningType(Enum):
    """Types of reasoning operations"""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


@dataclass
class KnowledgeEntry:
    """Represents a piece of knowledge"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: KnowledgeType = KnowledgeType.FACT
    content: str = ""
    subject: str = ""
    predicate: Optional[str] = None
    object: Optional[str] = None
    confidence: float = 1.0
    source: str = "user"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
    embedding: Optional[np.ndarray] = None
    related_entries: List[str] = field(default_factory=list)
    context: Optional[str] = None
    validity_period: Optional[timedelta] = None
    expires_at: Optional[datetime] = None


@dataclass
class Relationship:
    """Represents a relationship between knowledge entries"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: str = ""
    strength: float = 1.0
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Query:
    """Represents a knowledge query"""

    text: str
    type: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    include_reasoning: bool = False
    expand_relationships: bool = False
    min_confidence: float = 0.0


class KnowledgeBaseTool(BaseTool):
    """
    Advanced knowledge management tool with reasoning capabilities

    Features:
    - Semantic search with embeddings
    - Knowledge graph operations
    - Multiple reasoning types
    - Automatic relationship extraction
    - Knowledge validation and expiry
    - Incremental learning
    - Knowledge synthesis
    - Question answering
    """

    def __init__(self):
        metadata = ToolMetadata(
            name="knowledge_base",
            description="Advanced knowledge management with reasoning capabilities",
            category=ToolCategory.AI,
            version="2.0.0",
            tags=["knowledge", "reasoning", "graph", "semantic", "learning"],
            required_permissions=["storage_access", "compute_resources"],
            rate_limit=500,
            timeout=60,
            examples=[
                {
                    "description": "Store a fact",
                    "params": {
                        "action": "store",
                        "content": "Water boils at 100°C at sea level",
                        "type": "fact",
                        "tags": ["physics", "water", "temperature"],
                    },
                },
                {
                    "description": "Query knowledge",
                    "params": {
                        "action": "query",
                        "text": "What temperature does water boil?",
                        "include_reasoning": True,
                    },
                },
            ],
        )
        super().__init__(metadata)

        # Knowledge storage
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.relationships: Dict[str, Relationship] = {}

        # Knowledge graph
        self.graph = nx.DiGraph()

        # Embeddings
        self.embedding_model = None
        self.index = None  # FAISS index
        self.embedding_dim = 384

        # Indexes for fast lookup
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.type_index: Dict[KnowledgeType, Set[str]] = defaultdict(set)
        self.subject_index: Dict[str, Set[str]] = defaultdict(set)

        # Reasoning patterns
        self.reasoning_patterns = self._initialize_reasoning_patterns()

        # Statistics
        self.stats = {
            "total_entries": 0,
            "total_queries": 0,
            "successful_inferences": 0,
            "relationship_discoveries": 0,
        }

        # Storage
        self.storage_path = Path("./storage/knowledge_base")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize
        asyncio.create_task(self._initialize())

    async def _initialize(self):
        """Initialize the knowledge base"""
        try:
            # Load embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.embedding_dim)

            # Load persisted knowledge
            await self._load_knowledge()

            logger.info("Knowledge base initialized")

        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")

    async def _execute(self, **kwargs) -> Any:
        """Execute knowledge base operations"""
        action = kwargs.get("action", "").lower()

        if action == "store":
            return await self._store_knowledge(**kwargs)
        elif action == "query":
            return await self._query_knowledge(**kwargs)
        elif action == "update":
            return await self._update_knowledge(**kwargs)
        elif action == "delete":
            return await self._delete_knowledge(**kwargs)
        elif action == "relate":
            return await self._create_relationship(**kwargs)
        elif action == "reason":
            return await self._perform_reasoning(**kwargs)
        elif action == "validate":
            return await self._validate_knowledge(**kwargs)
        elif action == "synthesize":
            return await self._synthesize_knowledge(**kwargs)
        elif action == "explain":
            return await self._explain_reasoning(**kwargs)
        elif action == "stats":
            return self._get_statistics()
        elif action == "export":
            return await self._export_knowledge(**kwargs)
        elif action == "import":
            return await self._import_knowledge(**kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")

    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate knowledge base inputs"""
        action = kwargs.get("action")

        if not action:
            return False, "Action is required"

        if action == "store":
            if not kwargs.get("content"):
                return False, "Content is required for storing knowledge"

        elif action == "query":
            if not kwargs.get("text") and not kwargs.get("filters"):
                return False, "Either text or filters required for querying"

        elif action in ["update", "delete"]:
            if not kwargs.get("id"):
                return False, f"ID is required for {action}"

        elif action == "relate":
            if not kwargs.get("source_id") or not kwargs.get("target_id"):
                return False, "Source and target IDs required for relationships"

        return True, None

    async def _store_knowledge(self, **kwargs) -> Dict[str, Any]:
        """Store new knowledge entry"""
        # Create entry
        entry = KnowledgeEntry(
            type=KnowledgeType(kwargs.get("type", "fact").lower()),
            content=kwargs.get("content"),
            subject=kwargs.get("subject", ""),
            predicate=kwargs.get("predicate"),
            object=kwargs.get("object"),
            confidence=kwargs.get("confidence", 1.0),
            source=kwargs.get("source", "user"),
            tags=kwargs.get("tags", []),
            metadata=kwargs.get("metadata", {}),
            context=kwargs.get("context"),
            validity_period=kwargs.get("validity_period"),
        )

        # Extract subject if not provided
        if not entry.subject:
            entry.subject = self._extract_subject(entry.content)

        # Set expiry if validity period provided
        if entry.validity_period:
            entry.expires_at = datetime.now() + entry.validity_period

        # Generate embedding
        if self.embedding_model:
            embedding_text = f"{entry.subject} {entry.content}"
            entry.embedding = self.embedding_model.encode([embedding_text])[0]

            # Add to FAISS index
            if self.index:
                self.index.add(entry.embedding.reshape(1, -1))

        # Store entry
        self.entries[entry.id] = entry

        # Update indexes
        self._update_indexes(entry)

        # Add to graph
        self.graph.add_node(
            entry.id,
            **{
                "type": entry.type.value,
                "subject": entry.subject,
                "content": entry.content[:100],
            },
        )

        # Auto-discover relationships
        related = await self._discover_relationships(entry)
        entry.related_entries = [r["id"] for r in related[:5]]

        # Create relationships in graph
        for related_entry in related[:3]:
            self.graph.add_edge(
                entry.id,
                related_entry["id"],
                relation=related_entry.get("relation", "related_to"),
                weight=related_entry.get("score", 0.5),
            )

        # Persist
        await self._save_entry(entry)

        # Update stats
        self.stats["total_entries"] += 1

        return {
            "id": entry.id,
            "type": entry.type.value,
            "subject": entry.subject,
            "related_entries": len(entry.related_entries),
            "confidence": entry.confidence,
        }

    async def _query_knowledge(self, **kwargs) -> List[Dict[str, Any]]:
        """Query knowledge base"""
        query = Query(
            text=kwargs.get("text", ""),
            type=kwargs.get("type"),
            filters=kwargs.get("filters", {}),
            limit=kwargs.get("limit", 10),
            include_reasoning=kwargs.get("include_reasoning", False),
            expand_relationships=kwargs.get("expand_relationships", False),
            min_confidence=kwargs.get("min_confidence", 0.0),
        )

        # Update stats
        self.stats["total_queries"] += 1

        results = []

        if query.text:
            # Semantic search
            results = await self._semantic_search(query)
        else:
            # Filter-based search
            results = await self._filter_search(query)

        # Apply confidence filter
        results = [r for r in results if r.get("confidence", 0) >= query.min_confidence]

        # Expand relationships if requested
        if query.expand_relationships:
            for result in results:
                result["relationships"] = await self._get_relationships(result["id"])

        # Include reasoning if requested
        if query.include_reasoning:
            for result in results:
                result["reasoning"] = await self._generate_reasoning(
                    result["id"], query.text
                )

        # Sort by relevance
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Limit results
        results = results[: query.limit]

        # Update access counts
        for result in results:
            if result["id"] in self.entries:
                entry = self.entries[result["id"]]
                entry.accessed_count += 1
                entry.last_accessed = datetime.now()

        return results

    async def _semantic_search(self, query: Query) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings"""
        if not self.embedding_model or not self.index:
            return await self._fallback_search(query)

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query.text])[0]

        # Search in FAISS
        k = min(query.limit * 2, self.index.ntotal)
        if k == 0:
            return []

        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)

        results = []
        entry_list = list(self.entries.values())

        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(entry_list):
                entry = entry_list[idx]

                # Apply type filter
                if query.type and entry.type.value != query.type:
                    continue

                # Apply other filters
                if not self._matches_filters(entry, query.filters):
                    continue

                # Calculate relevance score
                score = 1.0 / (1.0 + distance)

                results.append(
                    {
                        "id": entry.id,
                        "type": entry.type.value,
                        "content": entry.content,
                        "subject": entry.subject,
                        "confidence": entry.confidence,
                        "score": score,
                        "source": entry.source,
                        "tags": entry.tags,
                        "created_at": entry.created_at.isoformat(),
                    }
                )

        return results

    async def _fallback_search(self, query: Query) -> List[Dict[str, Any]]:
        """Fallback text search when embeddings not available"""
        results = []
        query_lower = query.text.lower()

        for entry in self.entries.values():
            # Simple text matching
            if (
                query_lower in entry.content.lower()
                or query_lower in entry.subject.lower()
                or any(query_lower in tag for tag in entry.tags)
            ):

                # Apply filters
                if query.type and entry.type.value != query.type:
                    continue

                if not self._matches_filters(entry, query.filters):
                    continue

                # Calculate simple relevance score
                score = 0.5
                if query_lower in entry.subject.lower():
                    score += 0.3
                if query_lower in entry.content.lower():
                    score += 0.2

                results.append(
                    {
                        "id": entry.id,
                        "type": entry.type.value,
                        "content": entry.content,
                        "subject": entry.subject,
                        "confidence": entry.confidence,
                        "score": score,
                        "source": entry.source,
                        "tags": entry.tags,
                        "created_at": entry.created_at.isoformat(),
                    }
                )

        return results

    async def _filter_search(self, query: Query) -> List[Dict[str, Any]]:
        """Search using filters only"""
        results = []

        for entry in self.entries.values():
            # Apply type filter
            if query.type and entry.type.value != query.type:
                continue

            # Apply other filters
            if not self._matches_filters(entry, query.filters):
                continue

            results.append(
                {
                    "id": entry.id,
                    "type": entry.type.value,
                    "content": entry.content,
                    "subject": entry.subject,
                    "confidence": entry.confidence,
                    "score": entry.confidence,
                    "source": entry.source,
                    "tags": entry.tags,
                    "created_at": entry.created_at.isoformat(),
                }
            )

        return results

    def _matches_filters(self, entry: KnowledgeEntry, filters: Dict[str, Any]) -> bool:
        """Check if entry matches filters"""
        for key, value in filters.items():
            if key == "tags":
                if not any(tag in entry.tags for tag in value):
                    return False
            elif key == "source":
                if entry.source != value:
                    return False
            elif key == "min_confidence":
                if entry.confidence < value:
                    return False
            elif key == "subject":
                if value.lower() not in entry.subject.lower():
                    return False
            elif key == "after":
                if entry.created_at < value:
                    return False
            elif key == "before":
                if entry.created_at > value:
                    return False

        return True

    async def _create_relationship(self, **kwargs) -> Dict[str, Any]:
        """Create relationship between knowledge entries"""
        source_id = kwargs.get("source_id")
        target_id = kwargs.get("target_id")
        relation_type = kwargs.get("relation_type", "related_to")

        if source_id not in self.entries or target_id not in self.entries:
            raise ValueError("Source or target entry not found")

        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=kwargs.get("strength", 1.0),
            bidirectional=kwargs.get("bidirectional", False),
            metadata=kwargs.get("metadata", {}),
        )

        # Store relationship
        self.relationships[relationship.id] = relationship

        # Update graph
        self.graph.add_edge(
            source_id, target_id, relation=relation_type, weight=relationship.strength
        )

        if relationship.bidirectional:
            self.graph.add_edge(
                target_id,
                source_id,
                relation=relation_type,
                weight=relationship.strength,
            )

        # Update related entries
        self.entries[source_id].related_entries.append(target_id)
        if relationship.bidirectional:
            self.entries[target_id].related_entries.append(source_id)

        # Persist
        await self._save_relationship(relationship)

        self.stats["relationship_discoveries"] += 1

        return {
            "id": relationship.id,
            "source": self.entries[source_id].subject,
            "target": self.entries[target_id].subject,
            "relation": relation_type,
            "strength": relationship.strength,
        }

    async def _perform_reasoning(self, **kwargs) -> Dict[str, Any]:
        """Perform reasoning operations"""
        reasoning_type = ReasoningType(
            kwargs.get("reasoning_type", "deductive").lower()
        )
        premises = kwargs.get("premises", [])
        query = kwargs.get("query", "")

        if reasoning_type == ReasoningType.DEDUCTIVE:
            return await self._deductive_reasoning(premises, query)
        elif reasoning_type == ReasoningType.INDUCTIVE:
            return await self._inductive_reasoning(premises, query)
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            return await self._abductive_reasoning(premises, query)
        elif reasoning_type == ReasoningType.ANALOGICAL:
            return await self._analogical_reasoning(premises, query)
        elif reasoning_type == ReasoningType.CAUSAL:
            return await self._causal_reasoning(premises, query)
        else:
            raise ValueError(f"Unknown reasoning type: {reasoning_type}")

    async def _deductive_reasoning(
        self, premises: List[str], query: str
    ) -> Dict[str, Any]:
        """Perform deductive reasoning"""
        # Find relevant knowledge entries
        relevant_entries = []

        for premise in premises:
            results = await self._query_knowledge(text=premise, limit=5)
            relevant_entries.extend(results)

        # Apply reasoning patterns
        conclusions = []

        for pattern in self.reasoning_patterns["deductive"]:
            conclusion = self._apply_pattern(pattern, relevant_entries, query)
            if conclusion:
                conclusions.append(conclusion)

        if conclusions:
            self.stats["successful_inferences"] += 1

        return {
            "reasoning_type": "deductive",
            "premises_used": len(relevant_entries),
            "conclusions": conclusions,
            "confidence": (
                max([c["confidence"] for c in conclusions]) if conclusions else 0.0
            ),
        }

    async def _discover_relationships(
        self, entry: KnowledgeEntry
    ) -> List[Dict[str, Any]]:
        """Automatically discover relationships to other entries"""
        if not self.embedding_model or not entry.embedding:
            return []

        # Find similar entries
        similar = await self._semantic_search(
            Query(text=entry.content, limit=10, min_confidence=0.5)
        )

        # Filter out self
        similar = [s for s in similar if s["id"] != entry.id]

        # Analyze relationships
        relationships = []

        for sim_entry in similar:
            # Check for various relationship types
            relation = self._determine_relationship(
                entry, self.entries[sim_entry["id"]]
            )

            if relation:
                relationships.append(
                    {
                        "id": sim_entry["id"],
                        "relation": relation,
                        "score": sim_entry["score"],
                    }
                )

        return relationships

    def _determine_relationship(
        self, entry1: KnowledgeEntry, entry2: KnowledgeEntry
    ) -> Optional[str]:
        """Determine relationship type between two entries"""
        # Check for common patterns
        content1_lower = entry1.content.lower()
        content2_lower = entry2.content.lower()

        # Is-a relationship
        if (
            f"{entry1.subject} is a" in content2_lower
            or f"{entry2.subject} is a" in content1_lower
        ):
            return "is_a"

        # Part-of relationship
        if "part of" in content1_lower or "part of" in content2_lower:
            return "part_of"

        # Causes relationship
        if "causes" in content1_lower or "causes" in content2_lower:
            return "causes"

        # Prerequisites
        if "requires" in content1_lower or "requires" in content2_lower:
            return "requires"

        # Contradictions
        if self._check_contradiction(entry1, entry2):
            return "contradicts"

        # Default to related
        return "related_to"

    def _check_contradiction(
        self, entry1: KnowledgeEntry, entry2: KnowledgeEntry
    ) -> bool:
        """Check if two entries contradict each other"""
        # Simple contradiction detection
        negation_words = ["not", "never", "cannot", "won't", "doesn't", "isn't"]

        # Check if one negates the other
        for neg_word in negation_words:
            if (
                neg_word in entry1.content.lower()
                and entry1.subject in entry2.content
                and entry2.subject in entry1.content
            ):
                return True

        return False

    async def _synthesize_knowledge(self, **kwargs) -> Dict[str, Any]:
        """Synthesize new knowledge from existing entries"""
        topic = kwargs.get("topic", "")
        max_sources = kwargs.get("max_sources", 10)

        # Find relevant entries
        relevant = await self._query_knowledge(
            text=topic, limit=max_sources, expand_relationships=True
        )

        if not relevant:
            return {"error": "No relevant knowledge found"}

        # Extract key points
        key_points = []
        relationships = []

        for entry in relevant:
            key_points.append(
                {
                    "content": entry["content"],
                    "confidence": entry["confidence"],
                    "source": entry["source"],
                }
            )

            if "relationships" in entry:
                relationships.extend(entry["relationships"])

        # Generate synthesis
        synthesis = {
            "topic": topic,
            "summary": self._generate_summary(key_points),
            "key_facts": key_points[:5],
            "relationships_found": len(set(r["id"] for r in relationships)),
            "confidence": np.mean([kp["confidence"] for kp in key_points]),
            "sources_used": len(relevant),
        }

        # Store synthesis as new knowledge
        if kwargs.get("store_synthesis", True):
            await self._store_knowledge(
                content=synthesis["summary"],
                type="concept",
                subject=topic,
                source="synthesis",
                metadata={"synthesis": True, "sources": [e["id"] for e in relevant]},
                confidence=synthesis["confidence"],
            )

        return synthesis

    def _generate_summary(self, key_points: List[Dict[str, Any]]) -> str:
        """Generate summary from key points"""
        # Simple summarization
        if not key_points:
            return "No information available."

        # Sort by confidence
        key_points.sort(key=lambda x: x["confidence"], reverse=True)

        # Take top points
        top_points = key_points[:3]

        summary = "Based on available knowledge: "
        summary += " ".join([p["content"] for p in top_points])

        return summary

    async def _validate_knowledge(self, **kwargs) -> Dict[str, Any]:
        """Validate knowledge entries"""
        validation_results = {
            "total_entries": len(self.entries),
            "valid_entries": 0,
            "expired_entries": 0,
            "low_confidence_entries": 0,
            "contradictions_found": 0,
            "issues": [],
        }

        now = datetime.now()

        for entry_id, entry in self.entries.items():
            issues = []

            # Check expiry
            if entry.expires_at and entry.expires_at < now:
                issues.append("expired")
                validation_results["expired_entries"] += 1

            # Check confidence
            if entry.confidence < 0.5:
                issues.append("low_confidence")
                validation_results["low_confidence_entries"] += 1

            # Check for contradictions
            contradictions = await self._find_contradictions(entry)
            if contradictions:
                issues.append(f"contradicts: {len(contradictions)} entries")
                validation_results["contradictions_found"] += len(contradictions)

            if issues:
                validation_results["issues"].append(
                    {"id": entry_id, "subject": entry.subject, "issues": issues}
                )
            else:
                validation_results["valid_entries"] += 1

        return validation_results

    async def _find_contradictions(self, entry: KnowledgeEntry) -> List[str]:
        """Find entries that contradict the given entry"""
        contradictions = []

        # Use graph to find related entries
        if entry.id in self.graph:
            neighbors = list(self.graph.neighbors(entry.id))

            for neighbor_id in neighbors:
                neighbor = self.entries.get(neighbor_id)
                if neighbor and self._check_contradiction(entry, neighbor):
                    contradictions.append(neighbor_id)

        return contradictions

    def _extract_subject(self, content: str) -> str:
        """Extract subject from content"""
        # Simple extraction - take first noun phrase
        words = content.split()
        if words:
            # Look for common patterns
            for i, word in enumerate(words):
                if word.lower() in ["is", "are", "was", "were", "can", "will"]:
                    if i > 0:
                        return " ".join(words[:i])

            # Default to first few words
            return " ".join(words[:3])

        return "unknown"

    def _update_indexes(self, entry: KnowledgeEntry):
        """Update internal indexes"""
        # Tag index
        for tag in entry.tags:
            self.tag_index[tag].add(entry.id)

        # Type index
        self.type_index[entry.type].add(entry.id)

        # Subject index
        self.subject_index[entry.subject.lower()].add(entry.id)

    def _initialize_reasoning_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize reasoning patterns"""
        return {
            "deductive": [
                {
                    "name": "modus_ponens",
                    "pattern": "If A then B, A is true",
                    "conclusion": "B is true",
                },
                {
                    "name": "syllogism",
                    "pattern": "All A are B, X is A",
                    "conclusion": "X is B",
                },
            ],
            "inductive": [
                {
                    "name": "generalization",
                    "pattern": "Multiple instances of A have property B",
                    "conclusion": "All A likely have property B",
                }
            ],
            "abductive": [
                {
                    "name": "best_explanation",
                    "pattern": "B is observed, A explains B",
                    "conclusion": "A is likely true",
                }
            ],
        }

    def _apply_pattern(
        self, pattern: Dict[str, Any], entries: List[Dict[str, Any]], query: str
    ) -> Optional[Dict[str, Any]]:
        """Apply reasoning pattern to entries"""
        # Simplified pattern matching
        # In a real implementation, this would use more sophisticated logic

        if pattern["name"] == "modus_ponens":
            # Look for if-then statements
            for entry in entries:
                if (
                    "if" in entry["content"].lower()
                    and "then" in entry["content"].lower()
                ):
                    # Check if antecedent is satisfied
                    parts = entry["content"].lower().split("then")
                    if len(parts) == 2:
                        antecedent = parts[0].replace("if", "").strip()
                        consequent = parts[1].strip()

                        # Check if we have evidence for antecedent
                        for other_entry in entries:
                            if antecedent in other_entry["content"].lower():
                                return {
                                    "conclusion": consequent,
                                    "confidence": min(
                                        entry["confidence"], other_entry["confidence"]
                                    ),
                                    "pattern": "modus_ponens",
                                    "evidence": [entry["id"], other_entry["id"]],
                                }

        return None

    async def _generate_reasoning(self, entry_id: str, query: str) -> Dict[str, Any]:
        """Generate reasoning explanation for an entry"""
        if entry_id not in self.entries:
            return {"error": "Entry not found"}

        entry = self.entries[entry_id]

        # Build reasoning chain
        reasoning = {
            "entry_id": entry_id,
            "relevance": (
                "Direct match"
                if query.lower() in entry.content.lower()
                else "Semantic similarity"
            ),
            "confidence_factors": [],
        }

        # Analyze confidence factors
        if entry.source == "user":
            reasoning["confidence_factors"].append("User-provided knowledge")
        elif entry.source == "synthesis":
            reasoning["confidence_factors"].append("Synthesized from multiple sources")

        if entry.accessed_count > 10:
            reasoning["confidence_factors"].append(
                f"Frequently accessed ({entry.accessed_count} times)"
            )

        if entry.related_entries:
            reasoning["confidence_factors"].append(
                f"Connected to {len(entry.related_entries)} other entries"
            )

        # Check for supporting evidence
        supporting = []
        for related_id in entry.related_entries[:3]:
            if related_id in self.entries:
                related = self.entries[related_id]
                supporting.append(
                    {"content": related.content[:100], "confidence": related.confidence}
                )

        if supporting:
            reasoning["supporting_evidence"] = supporting

        return reasoning

    async def _get_relationships(self, entry_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entry"""
        relationships = []

        # Direct relationships from graph
        if entry_id in self.graph:
            for target_id in self.graph.neighbors(entry_id):
                edge_data = self.graph.get_edge_data(entry_id, target_id)

                if target_id in self.entries:
                    target = self.entries[target_id]
                    relationships.append(
                        {
                            "id": target_id,
                            "subject": target.subject,
                            "relation": edge_data.get("relation", "related_to"),
                            "strength": edge_data.get("weight", 1.0),
                        }
                    )

        # Relationships where this entry is target
        for source_id in self.graph.predecessors(entry_id):
            if source_id != entry_id and source_id in self.entries:
                edge_data = self.graph.get_edge_data(source_id, entry_id)
                source = self.entries[source_id]

                relationships.append(
                    {
                        "id": source_id,
                        "subject": source.subject,
                        "relation": f"inverse_{edge_data.get('relation', 'related_to')}",
                        "strength": edge_data.get("weight", 1.0),
                    }
                )

        return relationships

    def _get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        type_distribution = {}
        for k_type in KnowledgeType:
            type_distribution[k_type.value] = len(self.type_index[k_type])

        # Calculate average confidence
        confidences = [e.confidence for e in self.entries.values()]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Graph statistics
        graph_stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": (
                nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
            ),
            "connected_components": nx.number_weakly_connected_components(self.graph),
        }

        return {
            "total_entries": self.stats["total_entries"],
            "total_queries": self.stats["total_queries"],
            "successful_inferences": self.stats["successful_inferences"],
            "relationship_discoveries": self.stats["relationship_discoveries"],
            "type_distribution": type_distribution,
            "average_confidence": avg_confidence,
            "total_tags": len(self.tag_index),
            "graph_statistics": graph_stats,
            "storage_size_mb": self._calculate_storage_size(),
        }

    def _calculate_storage_size(self) -> float:
        """Calculate approximate storage size in MB"""
        # Rough estimation
        total_size = 0

        # Entries
        for entry in self.entries.values():
            total_size += len(entry.content.encode())
            if entry.embedding is not None:
                total_size += entry.embedding.nbytes

        # Relationships and graph
        total_size += len(pickle.dumps(self.relationships))
        total_size += len(pickle.dumps(self.graph))

        return total_size / (1024 * 1024)

    async def _update_knowledge(self, **kwargs) -> Dict[str, Any]:
        """Update existing knowledge entry"""
        entry_id = kwargs.get("id")

        if entry_id not in self.entries:
            raise ValueError(f"Entry {entry_id} not found")

        entry = self.entries[entry_id]

        # Update allowed fields
        if "content" in kwargs:
            entry.content = kwargs["content"]
            # Regenerate embedding
            if self.embedding_model:
                entry.embedding = self.embedding_model.encode([entry.content])[0]

        if "confidence" in kwargs:
            entry.confidence = kwargs["confidence"]

        if "tags" in kwargs:
            # Remove from old tag indexes
            for tag in entry.tags:
                self.tag_index[tag].discard(entry_id)
            # Update tags
            entry.tags = kwargs["tags"]
            # Add to new tag indexes
            for tag in entry.tags:
                self.tag_index[tag].add(entry_id)

        if "metadata" in kwargs:
            entry.metadata.update(kwargs["metadata"])

        entry.updated_at = datetime.now()

        # Persist
        await self._save_entry(entry)

        return {
            "id": entry_id,
            "updated": True,
            "updated_fields": [k for k in kwargs.keys() if k != "id"],
        }

    async def _delete_knowledge(self, **kwargs) -> Dict[str, Any]:
        """Delete knowledge entry"""
        entry_id = kwargs.get("id")

        if entry_id not in self.entries:
            raise ValueError(f"Entry {entry_id} not found")

        entry = self.entries[entry_id]

        # Remove from indexes
        for tag in entry.tags:
            self.tag_index[tag].discard(entry_id)
        self.type_index[entry.type].discard(entry_id)
        self.subject_index[entry.subject.lower()].discard(entry_id)

        # Remove from graph
        if entry_id in self.graph:
            self.graph.remove_node(entry_id)

        # Remove relationships
        to_remove = []
        for rel_id, rel in self.relationships.items():
            if rel.source_id == entry_id or rel.target_id == entry_id:
                to_remove.append(rel_id)

        for rel_id in to_remove:
            del self.relationships[rel_id]

        # Remove entry
        del self.entries[entry_id]

        # Remove from storage
        entry_file = self.storage_path / f"{entry_id}.json"
        if entry_file.exists():
            entry_file.unlink()

        self.stats["total_entries"] -= 1

        return {
            "id": entry_id,
            "deleted": True,
            "relationships_removed": len(to_remove),
        }

    async def _save_entry(self, entry: KnowledgeEntry):
        """Save entry to storage"""
        entry_file = self.storage_path / f"{entry.id}.json"

        entry_dict = {
            "id": entry.id,
            "type": entry.type.value,
            "content": entry.content,
            "subject": entry.subject,
            "predicate": entry.predicate,
            "object": entry.object,
            "confidence": entry.confidence,
            "source": entry.source,
            "tags": entry.tags,
            "metadata": entry.metadata,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "accessed_count": entry.accessed_count,
            "last_accessed": (
                entry.last_accessed.isoformat() if entry.last_accessed else None
            ),
            "related_entries": entry.related_entries,
            "context": entry.context,
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
        }

        async with aiofiles.open(entry_file, "w") as f:
            await f.write(json.dumps(entry_dict, indent=2))

        # Save embedding separately
        if entry.embedding is not None:
            embedding_file = self.storage_path / f"{entry.id}.npy"
            np.save(embedding_file, entry.embedding)

    async def _save_relationship(self, relationship: Relationship):
        """Save relationship to storage"""
        rel_file = self.storage_path / f"rel_{relationship.id}.json"

        rel_dict = {
            "id": relationship.id,
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
            "relation_type": relationship.relation_type,
            "strength": relationship.strength,
            "bidirectional": relationship.bidirectional,
            "metadata": relationship.metadata,
            "created_at": relationship.created_at.isoformat(),
        }

        async with aiofiles.open(rel_file, "w") as f:
            await f.write(json.dumps(rel_dict, indent=2))

    async def _load_knowledge(self):
        """Load knowledge from storage"""
        try:
            # Load entries
            for entry_file in self.storage_path.glob("*.json"):
                if entry_file.name.startswith("rel_"):
                    continue

                async with aiofiles.open(entry_file, "r") as f:
                    entry_dict = json.loads(await f.read())

                entry = KnowledgeEntry(
                    id=entry_dict["id"],
                    type=KnowledgeType(entry_dict["type"]),
                    content=entry_dict["content"],
                    subject=entry_dict["subject"],
                    predicate=entry_dict.get("predicate"),
                    object=entry_dict.get("object"),
                    confidence=entry_dict["confidence"],
                    source=entry_dict["source"],
                    tags=entry_dict["tags"],
                    metadata=entry_dict["metadata"],
                    created_at=datetime.fromisoformat(entry_dict["created_at"]),
                    updated_at=datetime.fromisoformat(entry_dict["updated_at"]),
                    accessed_count=entry_dict["accessed_count"],
                    related_entries=entry_dict["related_entries"],
                    context=entry_dict.get("context"),
                )

                if entry_dict.get("last_accessed"):
                    entry.last_accessed = datetime.fromisoformat(
                        entry_dict["last_accessed"]
                    )

                if entry_dict.get("expires_at"):
                    entry.expires_at = datetime.fromisoformat(entry_dict["expires_at"])

                # Load embedding
                embedding_file = self.storage_path / f"{entry.id}.npy"
                if embedding_file.exists():
                    entry.embedding = np.load(embedding_file)
                    # Add to FAISS index
                    if self.index:
                        self.index.add(entry.embedding.reshape(1, -1))

                self.entries[entry.id] = entry
                self._update_indexes(entry)

                # Add to graph
                self.graph.add_node(
                    entry.id,
                    **{
                        "type": entry.type.value,
                        "subject": entry.subject,
                        "content": entry.content[:100],
                    },
                )

            # Load relationships
            for rel_file in self.storage_path.glob("rel_*.json"):
                async with aiofiles.open(rel_file, "r") as f:
                    rel_dict = json.loads(await f.read())

                relationship = Relationship(
                    id=rel_dict["id"],
                    source_id=rel_dict["source_id"],
                    target_id=rel_dict["target_id"],
                    relation_type=rel_dict["relation_type"],
                    strength=rel_dict["strength"],
                    bidirectional=rel_dict["bidirectional"],
                    metadata=rel_dict["metadata"],
                    created_at=datetime.fromisoformat(rel_dict["created_at"]),
                )

                self.relationships[relationship.id] = relationship

                # Add to graph
                if (
                    relationship.source_id in self.entries
                    and relationship.target_id in self.entries
                ):
                    self.graph.add_edge(
                        relationship.source_id,
                        relationship.target_id,
                        relation=relationship.relation_type,
                        weight=relationship.strength,
                    )

                    if relationship.bidirectional:
                        self.graph.add_edge(
                            relationship.target_id,
                            relationship.source_id,
                            relation=relationship.relation_type,
                            weight=relationship.strength,
                        )

            self.stats["total_entries"] = len(self.entries)
            logger.info(f"Loaded {len(self.entries)} knowledge entries")

        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")

    async def _export_knowledge(self, **kwargs) -> Dict[str, Any]:
        """Export knowledge base"""
        format = kwargs.get("format", "json")
        include_embeddings = kwargs.get("include_embeddings", False)

        export_data = {
            "metadata": {
                "version": self.metadata.version,
                "export_date": datetime.now().isoformat(),
                "total_entries": len(self.entries),
                "total_relationships": len(self.relationships),
            },
            "entries": [],
            "relationships": [],
        }

        # Export entries
        for entry in self.entries.values():
            entry_data = {
                "id": entry.id,
                "type": entry.type.value,
                "content": entry.content,
                "subject": entry.subject,
                "confidence": entry.confidence,
                "source": entry.source,
                "tags": entry.tags,
                "metadata": entry.metadata,
                "created_at": entry.created_at.isoformat(),
            }

            if include_embeddings and entry.embedding is not None:
                entry_data["embedding"] = entry.embedding.tolist()

            export_data["entries"].append(entry_data)

        # Export relationships
        for rel in self.relationships.values():
            export_data["relationships"].append(
                {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relation_type": rel.relation_type,
                    "strength": rel.strength,
                    "bidirectional": rel.bidirectional,
                }
            )

        if format == "json":
            export_file = self.storage_path / "export.json"
            async with aiofiles.open(export_file, "w") as f:
                await f.write(json.dumps(export_data, indent=2))

            return {
                "format": "json",
                "file": str(export_file),
                "entries_exported": len(export_data["entries"]),
                "relationships_exported": len(export_data["relationships"]),
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def _import_knowledge(self, **kwargs) -> Dict[str, Any]:
        """Import knowledge base"""
        file_path = kwargs.get("file_path")
        merge = kwargs.get("merge", False)

        if not file_path:
            raise ValueError("file_path is required")

        import_file = Path(file_path)
        if not import_file.exists():
            raise ValueError(f"File not found: {file_path}")

        async with aiofiles.open(import_file, "r") as f:
            import_data = json.loads(await f.read())

        if not merge:
            # Clear existing knowledge
            self.entries.clear()
            self.relationships.clear()
            self.graph.clear()
            self.tag_index.clear()
            self.type_index.clear()
            self.subject_index.clear()
            if self.index:
                self.index.reset()

        imported_entries = 0
        imported_relationships = 0

        # Import entries
        for entry_data in import_data.get("entries", []):
            entry = KnowledgeEntry(
                id=entry_data.get("id", str(uuid.uuid4())),
                type=KnowledgeType(entry_data["type"]),
                content=entry_data["content"],
                subject=entry_data["subject"],
                confidence=entry_data.get("confidence", 1.0),
                source=entry_data.get("source", "import"),
                tags=entry_data.get("tags", []),
                metadata=entry_data.get("metadata", {}),
                created_at=datetime.fromisoformat(entry_data["created_at"]),
            )

            # Generate embedding if not provided
            if "embedding" in entry_data:
                entry.embedding = np.array(entry_data["embedding"])
            elif self.embedding_model:
                entry.embedding = self.embedding_model.encode([entry.content])[0]

            if entry.embedding is not None and self.index:
                self.index.add(entry.embedding.reshape(1, -1))

            self.entries[entry.id] = entry
            self._update_indexes(entry)

            # Add to graph
            self.graph.add_node(
                entry.id, **{"type": entry.type.value, "subject": entry.subject}
            )

            imported_entries += 1

        # Import relationships
        for rel_data in import_data.get("relationships", []):
            if (
                rel_data["source_id"] in self.entries
                and rel_data["target_id"] in self.entries
            ):

                relationship = Relationship(
                    source_id=rel_data["source_id"],
                    target_id=rel_data["target_id"],
                    relation_type=rel_data["relation_type"],
                    strength=rel_data.get("strength", 1.0),
                    bidirectional=rel_data.get("bidirectional", False),
                )

                self.relationships[relationship.id] = relationship

                # Add to graph
                self.graph.add_edge(
                    relationship.source_id,
                    relationship.target_id,
                    relation=relationship.relation_type,
                    weight=relationship.strength,
                )

                if relationship.bidirectional:
                    self.graph.add_edge(
                        relationship.target_id,
                        relationship.source_id,
                        relation=relationship.relation_type,
                        weight=relationship.strength,
                    )

                imported_relationships += 1

        self.stats["total_entries"] = len(self.entries)

        return {
            "entries_imported": imported_entries,
            "relationships_imported": imported_relationships,
            "merge_mode": merge,
            "total_entries": len(self.entries),
            "total_relationships": len(self.relationships),
        }

    async def _explain_reasoning(self, **kwargs) -> Dict[str, Any]:
        """Explain reasoning process for a query"""
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 5)

        # Perform query
        results = await self._query_knowledge(
            text=query, limit=limit, include_reasoning=True, expand_relationships=True
        )

        if not results:
            return {"explanation": "No relevant knowledge found for the query."}

        # Build explanation
        explanation = {
            "query": query,
            "reasoning_steps": [],
            "relevant_entries": len(results),
            "confidence": max([r.get("confidence", 0) for r in results]),
        }

        # Step 1: Query understanding
        explanation["reasoning_steps"].append(
            {
                "step": "Query Understanding",
                "description": f"Analyzed query: '{query}'",
                "extracted_concepts": self._extract_concepts(query),
            }
        )

        # Step 2: Knowledge retrieval
        explanation["reasoning_steps"].append(
            {
                "step": "Knowledge Retrieval",
                "description": f"Found {len(results)} relevant entries",
                "top_matches": [
                    {
                        "subject": r["subject"],
                        "relevance_score": r.get("score", 0),
                        "confidence": r["confidence"],
                    }
                    for r in results[:3]
                ],
            }
        )

        # Step 3: Relationship analysis
        all_relationships = []
        for result in results:
            if "relationships" in result:
                all_relationships.extend(result["relationships"])

        explanation["reasoning_steps"].append(
            {
                "step": "Relationship Analysis",
                "description": f"Analyzed {len(all_relationships)} relationships",
                "relationship_types": list(
                    set(r["relation"] for r in all_relationships)
                ),
            }
        )

        # Step 4: Inference
        inferences = []
        if len(results) > 1:
            # Look for patterns
            common_tags = set(results[0]["tags"])
            for r in results[1:]:
                common_tags &= set(r["tags"])

            if common_tags:
                inferences.append(f"Common themes: {', '.join(common_tags)}")

        explanation["reasoning_steps"].append(
            {
                "step": "Inference",
                "description": "Drew conclusions from available knowledge",
                "inferences": inferences
                or ["Direct knowledge available, no inference needed"],
            }
        )

        # Step 5: Answer synthesis
        answer = self._synthesize_answer(results, query)
        explanation["reasoning_steps"].append(
            {
                "step": "Answer Synthesis",
                "description": "Combined knowledge to form answer",
                "answer": answer,
            }
        )

        return explanation

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple concept extraction
        # Remove common words
        stop_words = {
            "the",
            "is",
            "at",
            "which",
            "on",
            "and",
            "a",
            "an",
            "as",
            "are",
            "was",
            "were",
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
        }

        words = text.lower().split()
        concepts = [w for w in words if w not in stop_words and len(w) > 2]

        return concepts

    def _synthesize_answer(self, results: List[Dict[str, Any]], query: str) -> str:
        """Synthesize answer from results"""
        if not results:
            return "No information available."

        # Take the most relevant result
        best_result = results[0]

        answer = f"Based on available knowledge: {best_result['content']}"

        # Add confidence qualifier
        if best_result["confidence"] < 0.7:
            answer += " (Note: Low confidence in this information)"
        elif best_result["confidence"] < 0.9:
            answer += " (Moderate confidence)"

        return answer

    async def _inductive_reasoning(
        self, premises: List[str], query: str
    ) -> Dict[str, Any]:
        """Perform inductive reasoning"""
        # Find examples matching the pattern
        examples = []

        for premise in premises:
            results = await self._query_knowledge(text=premise, limit=10)
            examples.extend(results)

        if len(examples) < 3:
            return {
                "reasoning_type": "inductive",
                "error": "Insufficient examples for inductive reasoning",
                "examples_found": len(examples),
            }

        # Look for patterns
        patterns = self._find_patterns(examples)

        # Generate generalizations
        generalizations = []
        for pattern in patterns:
            if pattern["frequency"] > 0.7:  # Pattern appears in >70% of examples
                generalizations.append(
                    {
                        "generalization": pattern["description"],
                        "confidence": pattern["frequency"],
                        "based_on": pattern["example_count"],
                    }
                )

        return {
            "reasoning_type": "inductive",
            "examples_analyzed": len(examples),
            "patterns_found": len(patterns),
            "generalizations": generalizations,
        }

    async def _abductive_reasoning(
        self, observations: List[str], query: str
    ) -> Dict[str, Any]:
        """Perform abductive reasoning (inference to best explanation)"""
        # Find potential explanations
        explanations = []

        for observation in observations:
            # Find entries that could explain the observation
            results = await self._query_knowledge(
                text=f"causes {observation} explains {observation}", limit=5
            )

            for result in results:
                explanations.append(
                    {
                        "explanation": result["content"],
                        "explains": observation,
                        "confidence": result["confidence"] * result.get("score", 0.5),
                    }
                )

        # Rank explanations
        explanations.sort(key=lambda x: x["confidence"], reverse=True)

        # Select best explanation
        best_explanation = explanations[0] if explanations else None

        return {
            "reasoning_type": "abductive",
            "observations": observations,
            "potential_explanations": len(explanations),
            "best_explanation": best_explanation,
            "alternative_explanations": (
                explanations[1:3] if len(explanations) > 1 else []
            ),
        }

    async def _analogical_reasoning(
        self, source_domain: List[str], target: str
    ) -> Dict[str, Any]:
        """Perform analogical reasoning"""
        # Find source domain knowledge
        source_knowledge = []
        for item in source_domain:
            results = await self._query_knowledge(text=item, limit=3)
            source_knowledge.extend(results)

        if not source_knowledge:
            return {
                "reasoning_type": "analogical",
                "error": "No knowledge found for source domain",
            }

        # Find similar structure in target domain
        target_results = await self._query_knowledge(text=target, limit=5)

        # Map relationships
        analogies = []
        for source in source_knowledge:
            for target_item in target_results:
                similarity = self._calculate_structural_similarity(source, target_item)
                if similarity > 0.6:
                    analogies.append(
                        {
                            "source": source["subject"],
                            "target": target_item["subject"],
                            "mapped_properties": self._map_properties(
                                source, target_item
                            ),
                            "similarity": similarity,
                        }
                    )

        return {
            "reasoning_type": "analogical",
            "source_domain_size": len(source_knowledge),
            "target_matches": len(target_results),
            "analogies_found": analogies[:3],  # Top 3 analogies
        }

    async def _causal_reasoning(self, causes: List[str], effect: str) -> Dict[str, Any]:
        """Perform causal reasoning"""
        # Find causal relationships
        causal_chains = []

        for cause in causes:
            # Look for direct causation
            results = await self._query_knowledge(
                text=f"{cause} causes leads to results in", limit=5
            )

            for result in results:
                # Check if it relates to the effect
                if effect.lower() in result["content"].lower():
                    causal_chains.append(
                        {
                            "cause": cause,
                            "effect": effect,
                            "mechanism": result["content"],
                            "confidence": result["confidence"],
                            "direct": True,
                        }
                    )
                else:
                    # Look for indirect causation
                    indirect = await self._find_causal_path(result["subject"], effect)
                    if indirect:
                        causal_chains.append(
                            {
                                "cause": cause,
                                "effect": effect,
                                "mechanism": f"{result['content']} -> {indirect}",
                                "confidence": result["confidence"] * 0.8,
                                "direct": False,
                            }
                        )

        return {
            "reasoning_type": "causal",
            "causes_analyzed": len(causes),
            "causal_chains_found": len(causal_chains),
            "causal_relationships": causal_chains,
        }

    def _find_patterns(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find patterns in examples"""
        patterns = []

        # Tag patterns
        tag_counts = defaultdict(int)
        for ex in examples:
            for tag in ex.get("tags", []):
                tag_counts[tag] += 1

        for tag, count in tag_counts.items():
            if count > len(examples) * 0.5:  # Tag appears in >50% of examples
                patterns.append(
                    {
                        "type": "tag_pattern",
                        "description": f"Commonly tagged with '{tag}'",
                        "frequency": count / len(examples),
                        "example_count": count,
                    }
                )

        # Subject patterns (simplified)
        subjects = [ex["subject"].lower() for ex in examples]
        common_words = defaultdict(int)
        for subject in subjects:
            for word in subject.split():
                if len(word) > 3:  # Skip short words
                    common_words[word] += 1

        for word, count in common_words.items():
            if count > len(examples) * 0.4:
                patterns.append(
                    {
                        "type": "subject_pattern",
                        "description": f"Subjects often contain '{word}'",
                        "frequency": count / len(examples),
                        "example_count": count,
                    }
                )

        return patterns

    def _calculate_structural_similarity(
        self, entry1: Dict[str, Any], entry2: Dict[str, Any]
    ) -> float:
        """Calculate structural similarity between entries"""
        similarity = 0.0

        # Type similarity
        if entry1.get("type") == entry2.get("type"):
            similarity += 0.2

        # Tag overlap
        tags1 = set(entry1.get("tags", []))
        tags2 = set(entry2.get("tags", []))
        if tags1 and tags2:
            overlap = len(tags1 & tags2) / len(tags1 | tags2)
            similarity += 0.3 * overlap

        # Relationship similarity (simplified)
        if "relationships" in entry1 and "relationships" in entry2:
            rel_types1 = set(r["relation"] for r in entry1["relationships"])
            rel_types2 = set(r["relation"] for r in entry2["relationships"])
            if rel_types1 and rel_types2:
                rel_overlap = len(rel_types1 & rel_types2) / len(
                    rel_types1 | rel_types2
                )
                similarity += 0.3 * rel_overlap

        # Content similarity (simple word overlap)
        words1 = set(entry1["content"].lower().split())
        words2 = set(entry2["content"].lower().split())
        if words1 and words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
            similarity += 0.2 * word_overlap

        return similarity

    def _map_properties(
        self, source: Dict[str, Any], target: Dict[str, Any]
    ) -> List[str]:
        """Map properties from source to target in analogy"""
        mapped = []

        # Simple property mapping based on relationships
        if "relationships" in source:
            for rel in source["relationships"]:
                mapped.append(f"{rel['relation']} (from source)")

        return mapped

    async def _find_causal_path(self, intermediate: str, effect: str) -> Optional[str]:
        """Find causal path from intermediate to effect"""
        # Simplified path finding
        results = await self._query_knowledge(text=f"{intermediate} {effect}", limit=3)

        for result in results:
            if effect.lower() in result["content"].lower():
                return result["content"]

        return None

    def _get_parameter_documentation(self) -> Dict[str, Any]:
        """Get parameter documentation for the knowledge base"""
        return {
            "action": {
                "type": "string",
                "required": True,
                "enum": [
                    "store",
                    "query",
                    "update",
                    "delete",
                    "relate",
                    "reason",
                    "validate",
                    "synthesize",
                    "explain",
                    "stats",
                    "export",
                    "import",
                ],
                "description": "Action to perform",
            },
            "content": {
                "type": "string",
                "required": "for store action",
                "description": "Knowledge content to store",
            },
            "type": {
                "type": "string",
                "required": False,
                "enum": [
                    "fact",
                    "concept",
                    "procedure",
                    "rule",
                    "experience",
                    "reference",
                    "relationship",
                ],
                "description": "Type of knowledge entry",
            },
            "text": {
                "type": "string",
                "required": "for query action",
                "description": "Query text for searching",
            },
            "id": {
                "type": "string",
                "required": "for update, delete actions",
                "description": "ID of knowledge entry",
            },
            "source_id": {
                "type": "string",
                "required": "for relate action",
                "description": "Source entry ID for relationship",
            },
            "target_id": {
                "type": "string",
                "required": "for relate action",
                "description": "Target entry ID for relationship",
            },
            "reasoning_type": {
                "type": "string",
                "required": "for reason action",
                "enum": ["deductive", "inductive", "abductive", "analogical", "causal"],
                "description": "Type of reasoning to perform",
            },
            "tags": {
                "type": "list",
                "required": False,
                "description": "Tags for categorizing knowledge",
            },
            "confidence": {
                "type": "float",
                "required": False,
                "description": "Confidence level (0-1)",
            },
            "filters": {
                "type": "dict",
                "required": False,
                "description": "Filters for querying",
            },
        }
