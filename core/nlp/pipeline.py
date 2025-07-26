"""
Advanced NLP Pipeline
Handles intent recognition, entity extraction, and semantic analysis
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

from ..logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class NLPResult:
    """Results from NLP processing"""
    text: str
    intent: str
    confidence: float
    entities: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None
    sentiment: Optional[float] = None
    

class NLPPipeline:
    """Advanced NLP processing pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.spacy_model = None
        self.intent_classifier = None
        self.embedding_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """Initialize NLP models"""
        logger.info(f"Initializing NLP pipeline on {self.device}...")
        
        # Load models in parallel
        await asyncio.gather(
            self._load_spacy(),
            self._load_intent_classifier(),
            self._load_embedding_model()
        )
        
        logger.info("NLP pipeline ready")
        
    async def _load_spacy(self):
        """Load spaCy model for entity extraction"""
        try:
            self.spacy_model = spacy.load("en_core_web_sm")
        except:
            logger.info("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.spacy_model = spacy.load("en_core_web_sm")
            
    async def _load_intent_classifier(self):
        """Load intent classification model"""
        model_name = self.config.get("nlp.intent_model", "distilbert-base-uncased")
        self.intent_classifier = pipeline(
            "text-classification",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        
    async def _load_embedding_model(self):
        """Load sentence embedding model"""
        from sentence_transformers import SentenceTransformer
        model_name = self.config.get("nlp.embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        
    async def process(self, text: str) -> NLPResult:
        """Process text through the full NLP pipeline"""
        # Run all processing in parallel
        results = await asyncio.gather(
            self._extract_intent(text),
            self._extract_entities(text),
            self._generate_embeddings(text),
            self._analyze_sentiment(text)
        )
        
        intent, confidence = results[0]
        entities = results[1]
        embeddings = results[2]
        sentiment = results[3]
        
        return NLPResult(
            text=text,
            intent=intent,
            confidence=confidence,
            entities=entities,
            embeddings=embeddings,
            sentiment=sentiment
        )
        
    async def _extract_intent(self, text: str) -> tuple[str, float]:
        """Extract intent from text"""
        # Custom intent patterns first
        intent_patterns = {
            "weather": ["weather", "temperature", "forecast", "rain", "sunny"],
            "time": ["time", "clock", "hour", "minute", "when"],
            "reminder": ["remind", "reminder", "alert", "notification"],
            "search": ["search", "find", "look up", "google"],
            "music": ["play", "music", "song", "spotify", "pause"],
            "smart_home": ["light", "door", "temperature", "turn on", "turn off"],
            "system": ["shutdown", "restart", "sleep", "volume"],
        }
        
        text_lower = text.lower()
        for intent, keywords in intent_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return f"plugin:{intent}", 0.9
                
        # Fallback to ML classifier
        try:
            result = self.intent_classifier(text)[0]
            return result['label'], result['score']
        except:
            return "general", 0.5
            
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities and key information"""
        doc = self.spacy_model(text)
        
        entities = {
            "persons": [],
            "locations": [],
            "organizations": [],
            "dates": [],
            "times": [],
            "numbers": [],
            "custom": {}
        }
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["persons"].append(ent.text)
            elif ent.label_ in ["LOC", "GPE"]:
                entities["locations"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "TIME":
                entities["times"].append(ent.text)
            elif ent.label_ in ["CARDINAL", "ORDINAL", "QUANTITY"]:
                entities["numbers"].append(ent.text)
                
        # Extract custom patterns (e.g., email, phone)
        import re
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            entities["custom"]["emails"] = emails
            
        return entities
        
    async def _generate_embeddings(self, text: str) -> np.ndarray:
        """Generate sentence embeddings for semantic search"""
        embeddings = self.embedding_model.encode(text)
        return embeddings
        
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of the text"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0