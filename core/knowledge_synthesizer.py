"""
Knowledge Synthesizer for JARVIS
=================================

Advanced system for synthesizing, organizing, and retrieving knowledge.
"""

import asyncio
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge
import spacy
import nltk
from nltk.corpus import wordnet
import wikipedia
import requests
from bs4 import BeautifulSoup
import hashlib
import sqlite3
from collections import defaultdict, deque
import re

logger = get_logger(__name__)

# Metrics
knowledge_items_processed = Counter(
    "knowledge_items_processed_total", "Total knowledge items processed"
)
synthesis_operations = Counter(
    "synthesis_operations_total", "Total synthesis operations", ["type"]
)
knowledge_graph_size = Gauge(
    "knowledge_graph_nodes", "Number of nodes in knowledge graph"
)
retrieval_time = Histogram(
    "knowledge_retrieval_duration_seconds", "Knowledge retrieval time"
)


@dataclass
class KnowledgeItem:
    """Represents a single piece of knowledge"""

    id: str
    content: str
    source: str
    timestamp: datetime
    confidence: float = 1.0
    embeddings: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[str, str]] = field(
        default_factory=list
    )  # (relation_type, target_id)
    tags: Set[str] = field(default_factory=set)

    def __hash__(self):
        return hash(self.id)


@dataclass
class KnowledgeCluster:
    """Represents a cluster of related knowledge"""

    id: str
    items: List[KnowledgeItem]
    centroid: Optional[np.ndarray] = None
    summary: Optional[str] = None
    coherence_score: float = 0.0
    topics: List[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """Result of knowledge synthesis"""

    synthesized_content: str
    source_items: List[str]  # IDs of source knowledge items
    confidence: float
    synthesis_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeSynthesizer:
    """
    Advanced knowledge synthesis and management system.

    Features:
    - Multi-source knowledge integration
    - Semantic clustering and organization
    - Knowledge graph construction
    - Fact verification and validation
    - Automatic summarization
    - Question answering
    - Knowledge evolution tracking
    - Contradiction detection and resolution
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        qa_model: str = "deepset/roberta-base-squad2",
        db_path: Path = Path("./knowledge.db"),
        index_dimension: int = 384,
        enable_web_search: bool = True,
    ):

        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_dimension = index_dimension
        self.enable_web_search = enable_web_search
        self.db_path = db_path

        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model)

        # Initialize NLP tools
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Spacy model not found, some features will be limited")
            self.nlp = None

        # Knowledge storage
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.knowledge_graph = nx.DiGraph()
        self.clusters: Dict[str, KnowledgeCluster] = {}

        # Vector index for similarity search
        self.index = faiss.IndexFlatL2(index_dimension)
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}

        # Initialize database
        self._init_database()

        # Load existing knowledge
        self._load_knowledge()

        logger.info(
            "Knowledge Synthesizer initialized",
            embedding_model=embedding_model,
            index_size=self.index.ntotal,
        )

    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Knowledge items table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                timestamp REAL,
                confidence REAL,
                embeddings BLOB,
                metadata TEXT,
                tags TEXT
            )
        """
        )

        # Relations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                source_id TEXT,
                target_id TEXT,
                relation_type TEXT,
                confidence REAL,
                timestamp REAL,
                FOREIGN KEY (source_id) REFERENCES knowledge_items(id),
                FOREIGN KEY (target_id) REFERENCES knowledge_items(id)
            )
        """
        )

        # Synthesis history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS synthesis_history (
                id TEXT PRIMARY KEY,
                synthesized_content TEXT,
                source_items TEXT,
                synthesis_type TEXT,
                confidence REAL,
                timestamp REAL,
                metadata TEXT
            )
        """
        )

        # Create indices
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON knowledge_items(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON knowledge_items(source)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id)"
        )

        conn.commit()
        conn.close()

    def _load_knowledge(self):
        """Load existing knowledge from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load knowledge items
        cursor.execute("SELECT * FROM knowledge_items")
        for row in cursor.fetchall():
            item = KnowledgeItem(
                id=row[0],
                content=row[1],
                source=row[2],
                timestamp=datetime.fromtimestamp(row[3]),
                confidence=row[4],
                embeddings=pickle.loads(row[5]) if row[5] else None,
                metadata=json.loads(row[6]) if row[6] else {},
                tags=set(json.loads(row[7])) if row[7] else set(),
            )
            self.knowledge_items[item.id] = item

            # Add to graph
            self.knowledge_graph.add_node(item.id, item=item)

            # Add to index if embeddings exist
            if item.embeddings is not None:
                idx = self.index.ntotal
                self.index.add(item.embeddings.reshape(1, -1))
                self.id_to_index[item.id] = idx
                self.index_to_id[idx] = item.id

        # Load relations
        cursor.execute("SELECT * FROM relations")
        for row in cursor.fetchall():
            source_id, target_id, relation_type = row[0], row[1], row[2]
            if source_id in self.knowledge_items and target_id in self.knowledge_items:
                self.knowledge_graph.add_edge(
                    source_id, target_id, relation=relation_type, confidence=row[3]
                )
                self.knowledge_items[source_id].relations.append(
                    (relation_type, target_id)
                )

        conn.close()

        # Update metrics
        knowledge_graph_size.set(self.knowledge_graph.number_of_nodes())

        logger.info(f"Loaded {len(self.knowledge_items)} knowledge items")

    async def add_knowledge(
        self,
        content: str,
        source: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeItem:
        """Add new knowledge to the system"""
        # Generate ID
        item_id = hashlib.sha256(
            f"{content}{source}{datetime.now()}".encode()
        ).hexdigest()[:16]

        # Generate embeddings
        embeddings = self.embedding_model.encode(content)

        # Extract entities and tags
        tags = self._extract_tags(content)

        # Create knowledge item
        item = KnowledgeItem(
            id=item_id,
            content=content,
            source=source,
            timestamp=datetime.now(),
            confidence=confidence,
            embeddings=embeddings,
            metadata=metadata or {},
            tags=tags,
        )

        # Add to storage
        self.knowledge_items[item_id] = item
        self.knowledge_graph.add_node(item_id, item=item)

        # Add to index
        idx = self.index.ntotal
        self.index.add(embeddings.reshape(1, -1))
        self.id_to_index[item_id] = idx
        self.index_to_id[idx] = item_id

        # Find and add relations
        await self._find_relations(item)

        # Persist to database
        self._save_knowledge_item(item)

        # Update metrics
        knowledge_items_processed.inc()
        knowledge_graph_size.set(self.knowledge_graph.number_of_nodes())

        logger.info(f"Added knowledge item: {item_id}")

        return item

    def _extract_tags(self, content: str) -> Set[str]:
        """Extract tags from content"""
        tags = set()

        if self.nlp:
            doc = self.nlp(content)

            # Extract named entities
            for ent in doc.ents:
                tags.add(f"entity:{ent.label_}:{ent.text.lower()}")

            # Extract key noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:
                    tags.add(f"concept:{chunk.text.lower()}")

            # Extract important words (nouns and verbs)
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"] and not token.is_stop:
                    tags.add(f"word:{token.lemma_.lower()}")

        return tags

    async def _find_relations(self, item: KnowledgeItem):
        """Find relations between new item and existing knowledge"""
        # Find similar items
        similar_items = await self.find_similar(item.content, k=10, threshold=0.7)

        for similar_item, similarity in similar_items:
            if similar_item.id != item.id:
                # Determine relation type based on similarity and content analysis
                relation_type = self._determine_relation_type(
                    item, similar_item, similarity
                )

                if relation_type:
                    # Add bidirectional relation
                    self.knowledge_graph.add_edge(
                        item.id,
                        similar_item.id,
                        relation=relation_type,
                        confidence=similarity,
                    )
                    item.relations.append((relation_type, similar_item.id))

                    # Also add reverse relation
                    reverse_relation = self._get_reverse_relation(relation_type)
                    self.knowledge_graph.add_edge(
                        similar_item.id,
                        item.id,
                        relation=reverse_relation,
                        confidence=similarity,
                    )
                    similar_item.relations.append((reverse_relation, item.id))

    def _determine_relation_type(
        self, item1: KnowledgeItem, item2: KnowledgeItem, similarity: float
    ) -> Optional[str]:
        """Determine the type of relation between two knowledge items"""
        if similarity > 0.95:
            return "duplicate"
        elif similarity > 0.8:
            # Check for contradiction
            if self._check_contradiction(item1.content, item2.content):
                return "contradicts"
            else:
                return "similar"
        elif similarity > 0.6:
            # Check for various relation types
            if self._check_elaboration(item1.content, item2.content):
                return "elaborates"
            elif self._check_supports(item1.content, item2.content):
                return "supports"
            else:
                return "related"

        return None

    def _get_reverse_relation(self, relation_type: str) -> str:
        """Get reverse relation type"""
        reverse_map = {
            "elaborates": "elaborated_by",
            "supports": "supported_by",
            "contradicts": "contradicts",
            "similar": "similar",
            "related": "related",
            "duplicate": "duplicate",
        }
        return reverse_map.get(relation_type, relation_type)

    def _check_contradiction(self, text1: str, text2: str) -> bool:
        """Check if two texts contradict each other"""
        # Simple heuristic - check for negation patterns
        negation_patterns = [
            (r"is\s+not", r"is(?!\s+not)"),
            (r"cannot", r"can(?!not)"),
            (r"never", r"always"),
            (r"false", r"true"),
            (r"incorrect", r"correct"),
        ]

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        for neg_pattern, pos_pattern in negation_patterns:
            if (
                re.search(neg_pattern, text1_lower)
                and re.search(pos_pattern, text2_lower)
            ) or (
                re.search(pos_pattern, text1_lower)
                and re.search(neg_pattern, text2_lower)
            ):
                return True

        return False

    def _check_elaboration(self, text1: str, text2: str) -> bool:
        """Check if text2 elaborates on text1"""
        # Simple heuristic - check if text2 is longer and contains key terms from text1
        if len(text2) > len(text1) * 1.5:
            text1_terms = set(text1.lower().split())
            text2_terms = set(text2.lower().split())
            overlap = len(text1_terms & text2_terms) / len(text1_terms)
            return overlap > 0.5
        return False

    def _check_supports(self, text1: str, text2: str) -> bool:
        """Check if texts support each other"""
        support_phrases = [
            "therefore",
            "thus",
            "hence",
            "consequently",
            "supports",
            "confirms",
            "validates",
            "proves",
        ]

        text_combined = f"{text1} {text2}".lower()
        return any(phrase in text_combined for phrase in support_phrases)

    async def find_similar(
        self, query: str, k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[KnowledgeItem, float]]:
        """Find similar knowledge items"""
        if self.index.ntotal == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).reshape(1, -1)

        # Search in index
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)

        # Convert to knowledge items with similarity scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.index_to_id:
                item_id = self.index_to_id[idx]
                item = self.knowledge_items[item_id]
                # Convert L2 distance to similarity (0-1)
                similarity = 1 / (1 + dist)

                if similarity >= threshold:
                    results.append((item, similarity))

        return results

    async def synthesize(
        self, query: str, synthesis_type: str = "summary", max_sources: int = 10
    ) -> SynthesisResult:
        """Synthesize knowledge based on query"""
        synthesis_operations.labels(type=synthesis_type).inc()

        # Find relevant knowledge
        relevant_items = await self.find_similar(query, k=max_sources, threshold=0.5)

        if not relevant_items:
            return SynthesisResult(
                synthesized_content="No relevant knowledge found.",
                source_items=[],
                confidence=0.0,
                synthesis_type=synthesis_type,
            )

        # Extract source content
        source_contents = [item.content for item, _ in relevant_items]
        source_ids = [item.id for item, _ in relevant_items]

        # Perform synthesis based on type
        if synthesis_type == "summary":
            synthesized = await self._synthesize_summary(source_contents)
        elif synthesis_type == "answer":
            synthesized = await self._synthesize_answer(query, source_contents)
        elif synthesis_type == "explanation":
            synthesized = await self._synthesize_explanation(query, source_contents)
        elif synthesis_type == "comparison":
            synthesized = await self._synthesize_comparison(source_contents)
        else:
            synthesized = await self._synthesize_combined(source_contents)

        # Calculate confidence
        avg_confidence = np.mean(
            [item.confidence * score for item, score in relevant_items]
        )

        result = SynthesisResult(
            synthesized_content=synthesized,
            source_items=source_ids,
            confidence=avg_confidence,
            synthesis_type=synthesis_type,
            metadata={"num_sources": len(source_ids), "query": query},
        )

        # Save synthesis result
        self._save_synthesis_result(result)

        return result

    async def _synthesize_summary(self, contents: List[str]) -> str:
        """Create a summary of multiple contents"""
        # Combine contents
        combined = "\n\n".join(contents)

        # Simple extractive summarization
        sentences = []
        for content in contents:
            # Extract first and most important sentences
            sents = content.split(". ")
            if sents:
                sentences.extend(sents[:2])  # Take first 2 sentences from each

        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for sent in sentences:
            if sent not in seen and len(sent) > 20:
                seen.add(sent)
                unique_sentences.append(sent)

        # Create summary
        summary = ". ".join(unique_sentences[:5]) + "."

        return summary

    async def _synthesize_answer(self, question: str, contents: List[str]) -> str:
        """Answer a question based on contents"""
        # Use QA model
        context = " ".join(contents[:3])  # Limit context size

        inputs = self.tokenizer(
            question, context, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.qa_model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][answer_start:answer_end]
            )
        )

        if not answer or answer == "[CLS]":
            # Fallback to extractive answer
            for content in contents:
                if question.lower() in content.lower():
                    # Find sentence containing question terms
                    sentences = content.split(". ")
                    for sent in sentences:
                        if any(
                            term in sent.lower() for term in question.lower().split()
                        ):
                            return sent

            return "Unable to find a specific answer in the knowledge base."

        return answer

    async def _synthesize_explanation(self, topic: str, contents: List[str]) -> str:
        """Create an explanation of a topic"""
        explanation_parts = []

        # Introduction
        explanation_parts.append(f"Based on the available knowledge about {topic}:")

        # Main points
        for i, content in enumerate(contents[:3], 1):
            # Extract key point
            sentences = content.split(". ")
            if sentences:
                explanation_parts.append(f"{i}. {sentences[0]}.")

        # Synthesis
        if len(contents) > 3:
            explanation_parts.append(
                f"Additional {len(contents) - 3} related pieces of information are available."
            )

        return "\n".join(explanation_parts)

    async def _synthesize_comparison(self, contents: List[str]) -> str:
        """Compare and contrast multiple pieces of content"""
        if len(contents) < 2:
            return "Not enough content for comparison."

        comparison_parts = ["Comparing the available information:"]

        # Find commonalities
        all_words = [set(content.lower().split()) for content in contents]
        common_words = set.intersection(*all_words)
        important_common = [
            w
            for w in common_words
            if len(w) > 4 and w not in ["that", "this", "with", "from"]
        ]

        if important_common:
            comparison_parts.append(f"Common themes: {', '.join(important_common[:5])}")

        # Find differences
        unique_aspects = []
        for i, words in enumerate(all_words[:3]):
            unique = words - common_words
            important_unique = [w for w in unique if len(w) > 4][:3]
            if important_unique:
                unique_aspects.append(
                    f"Source {i+1} uniquely mentions: {', '.join(important_unique)}"
                )

        comparison_parts.extend(unique_aspects)

        return "\n".join(comparison_parts)

    async def _synthesize_combined(self, contents: List[str]) -> str:
        """Create a combined synthesis of contents"""
        # Simple combination with deduplication
        sentences = []
        for content in contents:
            sentences.extend(content.split(". "))

        # Remove near-duplicates
        unique_sentences = []
        for sent in sentences:
            if not any(
                self._sentence_similarity(sent, existing) > 0.8
                for existing in unique_sentences
            ):
                unique_sentences.append(sent)

        # Combine up to 5 sentences
        combined = ". ".join(unique_sentences[:5])
        if combined and not combined.endswith("."):
            combined += "."

        return combined

    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences"""
        # Simple word overlap similarity
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the knowledge base"""
        # Synthesize answer
        result = await self.synthesize(question, synthesis_type="answer")

        # Get supporting evidence
        evidence = []
        for item_id in result.source_items[:3]:
            if item_id in self.knowledge_items:
                item = self.knowledge_items[item_id]
                evidence.append(
                    {
                        "content": (
                            item.content[:200] + "..."
                            if len(item.content) > 200
                            else item.content
                        ),
                        "source": item.source,
                        "confidence": item.confidence,
                    }
                )

        return {
            "answer": result.synthesized_content,
            "confidence": result.confidence,
            "evidence": evidence,
            "sources_used": len(result.source_items),
        }

    async def expand_knowledge(
        self, topic: str, max_items: int = 10
    ) -> List[KnowledgeItem]:
        """Expand knowledge on a topic using web search"""
        if not self.enable_web_search:
            return []

        new_items = []

        try:
            # Wikipedia search
            wiki_results = wikipedia.search(topic, results=3)
            for result in wiki_results:
                try:
                    page = wikipedia.page(result)
                    content = page.summary[:1000]  # Limit content size

                    item = await self.add_knowledge(
                        content=content,
                        source=f"wikipedia:{page.url}",
                        metadata={
                            "title": page.title,
                            "categories": page.categories[:5],
                        },
                    )
                    new_items.append(item)
                except:
                    continue

            # Could add more sources (news APIs, academic papers, etc.)

        except Exception as e:
            logger.warning(f"Knowledge expansion failed: {e}")

        return new_items

    async def cluster_knowledge(
        self, min_cluster_size: int = 3
    ) -> Dict[str, KnowledgeCluster]:
        """Cluster knowledge items by similarity"""
        if len(self.knowledge_items) < min_cluster_size:
            return {}

        # Get all embeddings
        embeddings = []
        item_ids = []

        for item_id, item in self.knowledge_items.items():
            if item.embeddings is not None:
                embeddings.append(item.embeddings)
                item_ids.append(item_id)

        if not embeddings:
            return {}

        # Perform clustering
        embeddings_array = np.vstack(embeddings)
        clustering = DBSCAN(eps=0.3, min_samples=min_cluster_size, metric="cosine")
        labels = clustering.fit_predict(embeddings_array)

        # Create clusters
        clusters = defaultdict(list)
        for item_id, label in zip(item_ids, labels):
            if label != -1:  # -1 is noise
                clusters[label].append(self.knowledge_items[item_id])

        # Convert to KnowledgeCluster objects
        self.clusters = {}
        for label, items in clusters.items():
            cluster_id = f"cluster_{label}"

            # Calculate centroid
            cluster_embeddings = np.vstack([item.embeddings for item in items])
            centroid = np.mean(cluster_embeddings, axis=0)

            # Generate summary
            contents = [item.content for item in items]
            summary = await self._synthesize_summary(contents[:5])

            # Extract topics
            all_tags = set()
            for item in items:
                all_tags.update(item.tags)

            topics = [
                tag.split(":")[-1] for tag in all_tags if tag.startswith("concept:")
            ][:5]

            cluster = KnowledgeCluster(
                id=cluster_id,
                items=items,
                centroid=centroid,
                summary=summary,
                coherence_score=self._calculate_cluster_coherence(items),
                topics=topics,
            )

            self.clusters[cluster_id] = cluster

        logger.info(f"Created {len(self.clusters)} knowledge clusters")

        return self.clusters

    def _calculate_cluster_coherence(self, items: List[KnowledgeItem]) -> float:
        """Calculate coherence score for a cluster"""
        if len(items) < 2:
            return 1.0

        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i].embeddings is not None and items[j].embeddings is not None:
                    sim = np.dot(items[i].embeddings, items[j].embeddings) / (
                        np.linalg.norm(items[i].embeddings)
                        * np.linalg.norm(items[j].embeddings)
                    )
                    similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    async def detect_contradictions(
        self,
    ) -> List[Tuple[KnowledgeItem, KnowledgeItem, float]]:
        """Detect contradictions in the knowledge base"""
        contradictions = []

        # Check items marked as contradicting
        for edge in self.knowledge_graph.edges(data=True):
            if edge[2].get("relation") == "contradicts":
                item1 = self.knowledge_items[edge[0]]
                item2 = self.knowledge_items[edge[1]]
                confidence = edge[2].get("confidence", 0.5)
                contradictions.append((item1, item2, confidence))

        return contradictions

    async def resolve_contradiction(
        self, item1: KnowledgeItem, item2: KnowledgeItem
    ) -> Optional[KnowledgeItem]:
        """Attempt to resolve a contradiction between two items"""
        # Strategy 1: Check source reliability
        source_reliability = {
            "wikipedia": 0.8,
            "scientific_paper": 0.95,
            "news": 0.6,
            "user": 0.5,
        }

        reliability1 = source_reliability.get(item1.source.split(":")[0], 0.5)
        reliability2 = source_reliability.get(item2.source.split(":")[0], 0.5)

        # Strategy 2: Check recency
        recency_weight = 0.2
        time_diff = abs((item1.timestamp - item2.timestamp).total_seconds())
        recency_factor = 1.0 / (
            1.0 + time_diff / (365 * 24 * 3600)
        )  # Decay over a year

        # Strategy 3: Check consensus
        consensus1 = len([r for r, _ in item1.relations if r == "supports"])
        consensus2 = len([r for r, _ in item2.relations if r == "supports"])

        # Calculate scores
        score1 = reliability1 + recency_weight * recency_factor + consensus1 * 0.1
        score2 = reliability2 + recency_weight * (1 - recency_factor) + consensus2 * 0.1

        # Create resolution
        if score1 > score2:
            resolution_content = (
                f"Resolved: {item1.content} (Contradiction with: {item2.content})"
            )
            resolution_confidence = score1 / (score1 + score2)
        else:
            resolution_content = (
                f"Resolved: {item2.content} (Contradiction with: {item1.content})"
            )
            resolution_confidence = score2 / (score1 + score2)

        # Add resolved knowledge
        resolution = await self.add_knowledge(
            content=resolution_content,
            source="system:contradiction_resolution",
            confidence=resolution_confidence,
            metadata={
                "resolved_items": [item1.id, item2.id],
                "resolution_scores": {"item1": score1, "item2": score2},
            },
        )

        return resolution

    def get_knowledge_subgraph(self, item_id: str, depth: int = 2) -> nx.DiGraph:
        """Get subgraph around a knowledge item"""
        if item_id not in self.knowledge_graph:
            return nx.DiGraph()

        # Get all nodes within depth
        nodes = {item_id}
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.knowledge_graph.successors(node))
                new_nodes.update(self.knowledge_graph.predecessors(node))
            nodes.update(new_nodes)

        # Create subgraph
        subgraph = self.knowledge_graph.subgraph(nodes).copy()

        return subgraph

    def _save_knowledge_item(self, item: KnowledgeItem):
        """Save knowledge item to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO knowledge_items 
            (id, content, source, timestamp, confidence, embeddings, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                item.id,
                item.content,
                item.source,
                item.timestamp.timestamp(),
                item.confidence,
                pickle.dumps(item.embeddings) if item.embeddings is not None else None,
                json.dumps(item.metadata),
                json.dumps(list(item.tags)),
            ),
        )

        # Save relations
        for relation_type, target_id in item.relations:
            cursor.execute(
                """
                INSERT OR REPLACE INTO relations
                (source_id, target_id, relation_type, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    item.id,
                    target_id,
                    relation_type,
                    0.8,  # Default confidence
                    datetime.now().timestamp(),
                ),
            )

        conn.commit()
        conn.close()

    def _save_synthesis_result(self, result: SynthesisResult):
        """Save synthesis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        result_id = hashlib.sha256(
            f"{result.synthesized_content}{datetime.now()}".encode()
        ).hexdigest()[:16]

        cursor.execute(
            """
            INSERT INTO synthesis_history
            (id, synthesized_content, source_items, synthesis_type, confidence, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result_id,
                result.synthesized_content,
                json.dumps(result.source_items),
                result.synthesis_type,
                result.confidence,
                datetime.now().timestamp(),
                json.dumps(result.metadata),
            ),
        )

        conn.commit()
        conn.close()

    async def export_knowledge_graph(self, format: str = "graphml") -> str:
        """Export knowledge graph in various formats"""
        if format == "graphml":
            return nx.generate_graphml(self.knowledge_graph)
        elif format == "json":
            data = nx.node_link_data(self.knowledge_graph)
            return json.dumps(data, indent=2)
        elif format == "dot":
            return nx.nx_pydot.to_pydot(self.knowledge_graph).to_string()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        stats = {
            "total_items": len(self.knowledge_items),
            "total_relations": self.knowledge_graph.number_of_edges(),
            "unique_sources": len(
                set(item.source for item in self.knowledge_items.values())
            ),
            "clusters": len(self.clusters),
            "avg_confidence": np.mean(
                [item.confidence for item in self.knowledge_items.values()]
            ),
            "graph_density": nx.density(self.knowledge_graph),
            "connected_components": nx.number_weakly_connected_components(
                self.knowledge_graph
            ),
        }

        # Relation type distribution
        relation_counts = defaultdict(int)
        for _, _, data in self.knowledge_graph.edges(data=True):
            relation_counts[data.get("relation", "unknown")] += 1
        stats["relation_distribution"] = dict(relation_counts)

        # Tag distribution
        tag_counts = defaultdict(int)
        for item in self.knowledge_items.values():
            for tag in item.tags:
                tag_type = tag.split(":")[0]
                tag_counts[tag_type] += 1
        stats["tag_distribution"] = dict(tag_counts)

        return stats


# Example usage
async def example_usage():
    """Example of using the Knowledge Synthesizer"""
    synthesizer = KnowledgeSynthesizer()

    # Add some knowledge
    knowledge_items = [
        (
            "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms that can learn from data.",
            "textbook:ai_fundamentals",
        ),
        (
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            "textbook:deep_learning",
        ),
        (
            "Neural networks are computing systems inspired by biological neural networks in animal brains.",
            "encyclopedia:neural_networks",
        ),
        (
            "Artificial intelligence aims to create machines that can perform tasks requiring human intelligence.",
            "textbook:ai_fundamentals",
        ),
        (
            "Machine learning algorithms improve their performance through experience without being explicitly programmed.",
            "research_paper:ml_survey",
        ),
    ]

    print("Adding knowledge items...")
    for content, source in knowledge_items:
        await synthesizer.add_knowledge(content, source)

    # Find similar knowledge
    print("\nFinding similar knowledge to 'neural networks':")
    similar = await synthesizer.find_similar("neural networks", k=3)
    for item, score in similar:
        print(f"- {item.content[:80]}... (similarity: {score:.2f})")

    # Synthesize knowledge
    print("\nSynthesizing summary about machine learning:")
    result = await synthesizer.synthesize("machine learning", synthesis_type="summary")
    print(f"Summary: {result.synthesized_content}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Sources used: {len(result.source_items)}")

    # Answer a question
    print("\nAnswering question:")
    answer = await synthesizer.answer_question(
        "What is the relationship between deep learning and AI?"
    )
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['confidence']:.2f}")

    # Cluster knowledge
    print("\nClustering knowledge...")
    clusters = await synthesizer.cluster_knowledge(min_cluster_size=2)
    for cluster_id, cluster in clusters.items():
        print(f"\n{cluster_id}:")
        print(f"  Summary: {cluster.summary}")
        print(f"  Topics: {', '.join(cluster.topics)}")
        print(f"  Items: {len(cluster.items)}")
        print(f"  Coherence: {cluster.coherence_score:.2f}")

    # Get statistics
    print("\nKnowledge base statistics:")
    stats = synthesizer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(example_usage())
