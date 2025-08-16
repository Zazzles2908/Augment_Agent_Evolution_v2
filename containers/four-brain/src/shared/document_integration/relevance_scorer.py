"""
Relevance Scorer for Four-Brain System v2
Advanced relevance scoring for document-to-query matching

Created: 2025-07-30 AEST
Purpose: Score document relevance to user queries with multiple algorithms and machine learning
"""

import asyncio
import json
import logging
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis
import numpy as np
from collections import Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoringAlgorithm(Enum):
    """Relevance scoring algorithms"""
    TF_IDF = "tf_idf"
    BM25 = "bm25"
    COSINE_SIMILARITY = "cosine_similarity"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID_SCORING = "hybrid_scoring"
    NEURAL_RANKING = "neural_ranking"
    LEARNING_TO_RANK = "learning_to_rank"

class RelevanceFeature(Enum):
    """Features used in relevance scoring"""
    TERM_FREQUENCY = "term_frequency"
    INVERSE_DOCUMENT_FREQUENCY = "inverse_document_frequency"
    DOCUMENT_LENGTH = "document_length"
    QUERY_COVERAGE = "query_coverage"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    POSITIONAL_FEATURES = "positional_features"
    FRESHNESS = "freshness"
    AUTHORITY = "authority"
    USER_INTERACTION = "user_interaction"

@dataclass
class ScoringFeatures:
    """Feature vector for relevance scoring"""
    term_frequency: float
    inverse_document_frequency: float
    document_length_norm: float
    query_coverage: float
    semantic_similarity: float
    position_score: float
    freshness_score: float
    authority_score: float
    user_interaction_score: float
    metadata: Dict[str, Any]

@dataclass
class RelevanceScore:
    """Comprehensive relevance score"""
    document_id: str
    query_id: str
    algorithm: ScoringAlgorithm
    score: float
    confidence: float
    features: ScoringFeatures
    explanation: Dict[str, Any]
    computed_at: datetime

@dataclass
class ScoringRequest:
    """Request for relevance scoring"""
    request_id: str
    query: str
    query_embedding: Optional[List[float]]
    documents: List[Dict[str, Any]]
    algorithm: ScoringAlgorithm
    feature_weights: Dict[str, float]
    user_context: Optional[Dict[str, Any]]
    created_at: datetime

@dataclass
class ScoringResult:
    """Complete scoring result"""
    request_id: str
    query: str
    algorithm_used: ScoringAlgorithm
    scores: List[RelevanceScore]
    total_documents: int
    processing_time: float
    feature_importance: Dict[str, float]
    metadata: Dict[str, Any]
    computed_at: datetime

class RelevanceScorer:
    """
    Advanced relevance scoring system for document-query matching
    
    Features:
    - Multiple scoring algorithms (TF-IDF, BM25, semantic similarity)
    - Machine learning-based relevance prediction
    - Feature engineering and importance analysis
    - User interaction learning and personalization
    - Real-time scoring with performance optimization
    - Explainable relevance scoring with feature attribution
    - Adaptive algorithm selection based on query type
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/23"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Configuration
        self.config = {
            'default_algorithm': ScoringAlgorithm.HYBRID_SCORING,
            'tf_idf_smoothing': 1.0,
            'bm25_k1': 1.2,
            'bm25_b': 0.75,
            'semantic_weight': 0.4,
            'lexical_weight': 0.6,
            'freshness_decay_days': 365,
            'min_score_threshold': 0.1,
            'max_results': 100,
            'feature_cache_ttl': 3600
        }
        
        # Default feature weights
        self.default_feature_weights = {
            'term_frequency': 0.25,
            'inverse_document_frequency': 0.20,
            'document_length_norm': 0.10,
            'query_coverage': 0.15,
            'semantic_similarity': 0.20,
            'position_score': 0.05,
            'freshness_score': 0.03,
            'authority_score': 0.02,
            'user_interaction_score': 0.00  # Will be learned
        }
        
        # Scoring state
        self.document_corpus: Dict[str, Dict[str, Any]] = {}
        self.term_frequencies: Dict[str, Dict[str, int]] = {}
        self.document_frequencies: Dict[str, int] = {}
        self.total_documents: int = 0
        self.scoring_history: List[ScoringResult] = []
        
        # Machine learning components
        self.user_feedback: Dict[str, List[Dict[str, Any]]] = {}
        self.feature_importance: Dict[str, float] = {}
        
        # Performance metrics
        self.metrics = {
            'total_scoring_requests': 0,
            'average_scoring_time': 0.0,
            'algorithm_usage': {alg.value: 0 for alg in ScoringAlgorithm},
            'average_score': 0.0,
            'score_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'user_satisfaction': 0.0
        }
        
        logger.info("🎯 Relevance Scorer initialized")
    
    async def initialize(self):
        """Initialize Redis connection and scoring services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load document corpus and statistics
            await self._load_document_corpus()
            await self._compute_corpus_statistics()
            
            # Load user feedback and feature importance
            await self._load_user_feedback()
            await self._load_feature_importance()
            
            # Start background services
            asyncio.create_task(self._corpus_maintenance())
            asyncio.create_task(self._model_training())
            
            logger.info("✅ Relevance Scorer Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Relevance Scorer: {e}")
            raise
    
    async def score_relevance(self, query: str, documents: List[Dict[str, Any]],
                            algorithm: ScoringAlgorithm = None,
                            feature_weights: Optional[Dict[str, float]] = None,
                            user_context: Optional[Dict[str, Any]] = None,
                            query_embedding: Optional[List[float]] = None) -> ScoringResult:
        """Score document relevance for query"""
        try:
            start_time = time.time()
            
            # Generate request ID
            request_id = f"score_{int(time.time() * 1000)}_{len(self.scoring_history)}"
            
            # Set defaults
            algorithm = algorithm or self.config['default_algorithm']
            feature_weights = feature_weights or self.default_feature_weights
            
            # Create scoring request
            request = ScoringRequest(
                request_id=request_id,
                query=query,
                query_embedding=query_embedding,
                documents=documents,
                algorithm=algorithm,
                feature_weights=feature_weights,
                user_context=user_context,
                created_at=datetime.now()
            )
            
            # Generate query embedding if not provided
            if not query_embedding:
                query_embedding = await self._generate_query_embedding(query)
                request.query_embedding = query_embedding
            
            # Score documents using specified algorithm
            relevance_scores = await self._execute_scoring_algorithm(request)
            
            # Sort by relevance score
            relevance_scores.sort(key=lambda x: x.score, reverse=True)
            
            # Calculate feature importance for this query
            feature_importance = await self._calculate_feature_importance(relevance_scores)
            
            # Create scoring result
            processing_time = time.time() - start_time
            
            result = ScoringResult(
                request_id=request_id,
                query=query,
                algorithm_used=algorithm,
                scores=relevance_scores,
                total_documents=len(documents),
                processing_time=processing_time,
                feature_importance=feature_importance,
                metadata={
                    'user_context': user_context,
                    'feature_weights': feature_weights,
                    'query_length': len(query.split())
                },
                computed_at=datetime.now()
            )
            
            # Store result
            self.scoring_history.append(result)
            await self._store_scoring_result(result)
            
            # Update metrics
            self.metrics['total_scoring_requests'] += 1
            self.metrics['algorithm_usage'][algorithm.value] += 1
            self._update_average_scoring_time(processing_time)
            self._update_score_statistics(relevance_scores)
            
            logger.info(f"✅ Relevance scored: {len(relevance_scores)} documents for '{query[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"❌ Relevance scoring failed: {e}")
            raise
    
    async def _execute_scoring_algorithm(self, request: ScoringRequest) -> List[RelevanceScore]:
        """Execute specific scoring algorithm"""
        try:
            if request.algorithm == ScoringAlgorithm.TF_IDF:
                return await self._tf_idf_scoring(request)
            elif request.algorithm == ScoringAlgorithm.BM25:
                return await self._bm25_scoring(request)
            elif request.algorithm == ScoringAlgorithm.COSINE_SIMILARITY:
                return await self._cosine_similarity_scoring(request)
            elif request.algorithm == ScoringAlgorithm.SEMANTIC_SIMILARITY:
                return await self._semantic_similarity_scoring(request)
            elif request.algorithm == ScoringAlgorithm.HYBRID_SCORING:
                return await self._hybrid_scoring(request)
            elif request.algorithm == ScoringAlgorithm.NEURAL_RANKING:
                return await self._neural_ranking_scoring(request)
            else:
                # Default to hybrid scoring
                return await self._hybrid_scoring(request)
            
        except Exception as e:
            logger.error(f"❌ Scoring algorithm execution failed: {e}")
            return []
    
    async def _tf_idf_scoring(self, request: ScoringRequest) -> List[RelevanceScore]:
        """TF-IDF based relevance scoring"""
        try:
            scores = []
            query_terms = self._tokenize(request.query)
            
            for doc in request.documents:
                doc_id = doc.get('document_id', doc.get('id', ''))
                content = doc.get('content', '')
                doc_terms = self._tokenize(content)
                
                # Calculate TF-IDF score
                tf_idf_score = 0.0
                
                for term in query_terms:
                    # Term frequency
                    tf = doc_terms.count(term) / len(doc_terms) if doc_terms else 0
                    tf = math.log(1 + tf)  # Log normalization
                    
                    # Inverse document frequency
                    df = self.document_frequencies.get(term, 1)
                    idf = math.log(self.total_documents / df) if df > 0 else 0
                    
                    tf_idf_score += tf * idf
                
                # Normalize by query length
                tf_idf_score = tf_idf_score / len(query_terms) if query_terms else 0
                
                # Create features
                features = await self._extract_features(request.query, doc, tf_idf_score)
                
                # Create relevance score
                score = RelevanceScore(
                    document_id=doc_id,
                    query_id=request.request_id,
                    algorithm=ScoringAlgorithm.TF_IDF,
                    score=tf_idf_score,
                    confidence=min(1.0, tf_idf_score * 2),  # Simple confidence
                    features=features,
                    explanation={
                        'algorithm': 'TF-IDF',
                        'query_terms': query_terms,
                        'matched_terms': [t for t in query_terms if t in doc_terms]
                    },
                    computed_at=datetime.now()
                )
                
                scores.append(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ TF-IDF scoring failed: {e}")
            return []
    
    async def _bm25_scoring(self, request: ScoringRequest) -> List[RelevanceScore]:
        """BM25 based relevance scoring"""
        try:
            scores = []
            query_terms = self._tokenize(request.query)
            
            # Calculate average document length
            avg_doc_length = sum(len(self._tokenize(doc.get('content', ''))) 
                               for doc in request.documents) / len(request.documents)
            
            for doc in request.documents:
                doc_id = doc.get('document_id', doc.get('id', ''))
                content = doc.get('content', '')
                doc_terms = self._tokenize(content)
                doc_length = len(doc_terms)
                
                # Calculate BM25 score
                bm25_score = 0.0
                
                for term in query_terms:
                    # Term frequency in document
                    tf = doc_terms.count(term)
                    
                    # Inverse document frequency
                    df = self.document_frequencies.get(term, 1)
                    idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
                    
                    # BM25 formula
                    k1 = self.config['bm25_k1']
                    b = self.config['bm25_b']
                    
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                    
                    bm25_score += idf * (numerator / denominator)
                
                # Create features
                features = await self._extract_features(request.query, doc, bm25_score)
                
                # Create relevance score
                score = RelevanceScore(
                    document_id=doc_id,
                    query_id=request.request_id,
                    algorithm=ScoringAlgorithm.BM25,
                    score=bm25_score,
                    confidence=min(1.0, bm25_score / 10),  # Normalize confidence
                    features=features,
                    explanation={
                        'algorithm': 'BM25',
                        'k1': k1,
                        'b': b,
                        'avg_doc_length': avg_doc_length,
                        'doc_length': doc_length
                    },
                    computed_at=datetime.now()
                )
                
                scores.append(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ BM25 scoring failed: {e}")
            return []
    
    async def _semantic_similarity_scoring(self, request: ScoringRequest) -> List[RelevanceScore]:
        """Semantic similarity based scoring using embeddings"""
        try:
            scores = []
            
            if not request.query_embedding:
                logger.warning("No query embedding available for semantic scoring")
                return scores
            
            for doc in request.documents:
                doc_id = doc.get('document_id', doc.get('id', ''))
                doc_embedding = doc.get('embedding')
                
                if not doc_embedding:
                    # Generate embedding for document
                    doc_embedding = await self._generate_document_embedding(doc.get('content', ''))
                
                if doc_embedding:
                    # Calculate cosine similarity
                    similarity = self._calculate_cosine_similarity(
                        request.query_embedding, doc_embedding
                    )
                    
                    # Create features
                    features = await self._extract_features(request.query, doc, similarity)
                    
                    # Create relevance score
                    score = RelevanceScore(
                        document_id=doc_id,
                        query_id=request.request_id,
                        algorithm=ScoringAlgorithm.SEMANTIC_SIMILARITY,
                        score=similarity,
                        confidence=similarity,  # Similarity is already 0-1
                        features=features,
                        explanation={
                            'algorithm': 'Semantic Similarity',
                            'similarity_type': 'cosine',
                            'embedding_dimensions': len(request.query_embedding)
                        },
                        computed_at=datetime.now()
                    )
                    
                    scores.append(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ Semantic similarity scoring failed: {e}")
            return []
    
    async def _hybrid_scoring(self, request: ScoringRequest) -> List[RelevanceScore]:
        """Hybrid scoring combining multiple algorithms"""
        try:
            # Get scores from different algorithms
            bm25_scores = await self._bm25_scoring(request)
            semantic_scores = await self._semantic_similarity_scoring(request)
            
            # Combine scores
            combined_scores = []
            
            # Create lookup for semantic scores
            semantic_lookup = {score.document_id: score for score in semantic_scores}
            
            for bm25_score in bm25_scores:
                doc_id = bm25_score.document_id
                semantic_score = semantic_lookup.get(doc_id)
                
                # Combine lexical and semantic scores
                lexical_weight = self.config['lexical_weight']
                semantic_weight = self.config['semantic_weight']
                
                combined_score_value = (
                    bm25_score.score * lexical_weight +
                    (semantic_score.score if semantic_score else 0) * semantic_weight
                )
                
                # Normalize combined score
                combined_score_value = combined_score_value / (lexical_weight + semantic_weight)
                
                # Create combined features
                features = bm25_score.features
                if semantic_score:
                    features.semantic_similarity = semantic_score.score
                
                # Create hybrid relevance score
                score = RelevanceScore(
                    document_id=doc_id,
                    query_id=request.request_id,
                    algorithm=ScoringAlgorithm.HYBRID_SCORING,
                    score=combined_score_value,
                    confidence=min(1.0, combined_score_value * 1.5),
                    features=features,
                    explanation={
                        'algorithm': 'Hybrid (BM25 + Semantic)',
                        'lexical_weight': lexical_weight,
                        'semantic_weight': semantic_weight,
                        'bm25_score': bm25_score.score,
                        'semantic_score': semantic_score.score if semantic_score else 0
                    },
                    computed_at=datetime.now()
                )
                
                combined_scores.append(score)
            
            return combined_scores
            
        except Exception as e:
            logger.error(f"❌ Hybrid scoring failed: {e}")
            return []
    
    async def _neural_ranking_scoring(self, request: ScoringRequest) -> List[RelevanceScore]:
        """Neural ranking based scoring (placeholder for ML model)"""
        try:
            # This would implement a neural ranking model
            # For now, fall back to hybrid scoring
            return await self._hybrid_scoring(request)
            
        except Exception as e:
            logger.error(f"❌ Neural ranking scoring failed: {e}")
            return []
    
    async def _extract_features(self, query: str, document: Dict[str, Any], 
                              base_score: float) -> ScoringFeatures:
        """Extract comprehensive features for relevance scoring"""
        try:
            content = document.get('content', '')
            doc_terms = self._tokenize(content)
            query_terms = self._tokenize(query)
            
            # Term frequency features
            tf_scores = []
            for term in query_terms:
                tf = doc_terms.count(term) / len(doc_terms) if doc_terms else 0
                tf_scores.append(tf)
            
            term_frequency = sum(tf_scores) / len(tf_scores) if tf_scores else 0
            
            # Inverse document frequency
            idf_scores = []
            for term in query_terms:
                df = self.document_frequencies.get(term, 1)
                idf = math.log(self.total_documents / df) if df > 0 else 0
                idf_scores.append(idf)
            
            inverse_document_frequency = sum(idf_scores) / len(idf_scores) if idf_scores else 0
            
            # Document length normalization
            document_length_norm = 1.0 / (1.0 + math.log(len(doc_terms))) if doc_terms else 0
            
            # Query coverage
            matched_terms = set(query_terms).intersection(set(doc_terms))
            query_coverage = len(matched_terms) / len(query_terms) if query_terms else 0
            
            # Position score (based on chunk index if available)
            chunk_index = document.get('chunk_index', 0)
            position_score = 1.0 / (1.0 + chunk_index * 0.1)  # Earlier chunks get higher score
            
            # Freshness score
            freshness_score = await self._calculate_freshness_score(document)
            
            # Authority score (based on document metadata)
            authority_score = await self._calculate_authority_score(document)
            
            # User interaction score (placeholder)
            user_interaction_score = 0.0
            
            return ScoringFeatures(
                term_frequency=term_frequency,
                inverse_document_frequency=inverse_document_frequency,
                document_length_norm=document_length_norm,
                query_coverage=query_coverage,
                semantic_similarity=0.0,  # Will be set by semantic scoring
                position_score=position_score,
                freshness_score=freshness_score,
                authority_score=authority_score,
                user_interaction_score=user_interaction_score,
                metadata={
                    'base_score': base_score,
                    'doc_length': len(doc_terms),
                    'query_length': len(query_terms),
                    'matched_terms': len(matched_terms)
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return ScoringFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, {})
    
    async def _calculate_freshness_score(self, document: Dict[str, Any]) -> float:
        """Calculate freshness score based on document age"""
        try:
            doc_metadata = document.get('document_metadata', {})
            modified_time = doc_metadata.get('modified_time')
            
            if not modified_time:
                return 0.5  # Neutral score for unknown age
            
            if isinstance(modified_time, str):
                modified_time = datetime.fromisoformat(modified_time)
            
            # Calculate age in days
            age_days = (datetime.now() - modified_time).days
            
            # Exponential decay
            decay_rate = 1.0 / self.config['freshness_decay_days']
            freshness_score = math.exp(-decay_rate * age_days)
            
            return freshness_score
            
        except Exception as e:
            logger.error(f"❌ Freshness score calculation failed: {e}")
            return 0.5
    
    async def _calculate_authority_score(self, document: Dict[str, Any]) -> float:
        """Calculate authority score based on document metadata"""
        try:
            doc_metadata = document.get('document_metadata', {})
            
            # Simple authority scoring (would be more sophisticated in production)
            authority_score = 0.5  # Base score
            
            # Boost for certain document types
            doc_type = doc_metadata.get('document_type', '')
            if doc_type in ['specification', 'manual', 'guide']:
                authority_score += 0.3
            
            # Boost for larger documents (more comprehensive)
            file_size = doc_metadata.get('file_size', 0)
            if file_size > 100000:  # > 100KB
                authority_score += 0.2
            
            return min(1.0, authority_score)
            
        except Exception as e:
            logger.error(f"❌ Authority score calculation failed: {e}")
            return 0.5
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms"""
        try:
            # Simple tokenization (would use advanced NLP in production)
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            tokens = text.split()
            
            # Remove stop words (simple list)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            
            return tokens
            
        except Exception as e:
            logger.error(f"❌ Tokenization failed: {e}")
            return []
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Cosine similarity calculation failed: {e}")
            return 0.0
    
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query using Brain-1"""
        try:
            # This would integrate with Brain-1's embedding API
            # For now, generate mock embedding
            embedding = np.random.rand(2000).tolist()  # 2000 dimensions for Supabase compatibility
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Query embedding generation failed: {e}")
            return []
    
    async def _generate_document_embedding(self, content: str) -> List[float]:
        """Generate embedding for document content"""
        try:
            # This would integrate with Brain-1's embedding API
            # For now, generate mock embedding
            embedding = np.random.rand(2000).tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Document embedding generation failed: {e}")
            return []
    
    async def _calculate_feature_importance(self, scores: List[RelevanceScore]) -> Dict[str, float]:
        """Calculate feature importance for current scoring"""
        try:
            if not scores:
                return {}
            
            # Simple feature importance calculation
            # In production, this would use more sophisticated methods
            
            feature_correlations = {}
            score_values = [score.score for score in scores]
            
            for score in scores:
                features = score.features
                feature_dict = asdict(features)
                
                for feature_name, feature_value in feature_dict.items():
                    if isinstance(feature_value, (int, float)):
                        if feature_name not in feature_correlations:
                            feature_correlations[feature_name] = []
                        feature_correlations[feature_name].append((feature_value, score.score))
            
            # Calculate simple correlation
            importance = {}
            for feature_name, values in feature_correlations.items():
                if len(values) > 1:
                    feature_vals = [v[0] for v in values]
                    score_vals = [v[1] for v in values]
                    
                    # Simple correlation coefficient
                    correlation = np.corrcoef(feature_vals, score_vals)[0, 1]
                    importance[feature_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                else:
                    importance[feature_name] = 0.0
            
            return importance
            
        except Exception as e:
            logger.error(f"❌ Feature importance calculation failed: {e}")
            return {}
    
    async def _load_document_corpus(self):
        """Load document corpus from Redis"""
        try:
            if self.redis_client:
                # Load indexed documents
                keys = await self.redis_client.keys("indexed_doc:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        doc_dict = json.loads(data)
                        doc_id = doc_dict.get('document_id', '')
                        if doc_id:
                            self.document_corpus[doc_id] = doc_dict
                
                self.total_documents = len(self.document_corpus)
                logger.info(f"Loaded {self.total_documents} documents into corpus")
            
        except Exception as e:
            logger.error(f"❌ Failed to load document corpus: {e}")
    
    async def _compute_corpus_statistics(self):
        """Compute corpus-wide statistics for scoring"""
        try:
            # Compute document frequencies
            self.document_frequencies = {}
            
            for doc_id, doc_data in self.document_corpus.items():
                # This would process document content to build term frequencies
                # For now, use mock data
                pass
            
            logger.info(f"Computed statistics for {len(self.document_frequencies)} terms")
            
        except Exception as e:
            logger.error(f"❌ Failed to compute corpus statistics: {e}")
    
    async def _load_user_feedback(self):
        """Load user feedback for learning"""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys("user_feedback:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        feedback_data = json.loads(data)
                        user_id = key.split(':')[1]
                        self.user_feedback[user_id] = feedback_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load user feedback: {e}")
    
    async def _load_feature_importance(self):
        """Load learned feature importance"""
        try:
            if self.redis_client:
                data = await self.redis_client.get("feature_importance")
                if data:
                    self.feature_importance = json.loads(data)
            
        except Exception as e:
            logger.error(f"❌ Failed to load feature importance: {e}")
    
    async def _store_scoring_result(self, result: ScoringResult):
        """Store scoring result in Redis"""
        if self.redis_client:
            try:
                key = f"scoring_result:{result.request_id}"
                data = json.dumps(asdict(result), default=str)
                await self.redis_client.setex(key, 86400, data)  # 24 hour TTL
            except Exception as e:
                logger.error(f"Failed to store scoring result: {e}")
    
    async def _corpus_maintenance(self):
        """Maintain document corpus and statistics"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Reload corpus if needed
                await self._load_document_corpus()
                await self._compute_corpus_statistics()
                
            except Exception as e:
                logger.error(f"❌ Corpus maintenance error: {e}")
    
    async def _model_training(self):
        """Train relevance models based on user feedback"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Train models based on user feedback
                # Update feature importance
                # Optimize algorithm parameters
                
            except Exception as e:
                logger.error(f"❌ Model training error: {e}")
    
    def _update_average_scoring_time(self, scoring_time: float):
        """Update average scoring time metric"""
        if self.metrics['total_scoring_requests'] == 1:
            self.metrics['average_scoring_time'] = scoring_time
        else:
            alpha = 0.1
            self.metrics['average_scoring_time'] = (
                alpha * scoring_time + 
                (1 - alpha) * self.metrics['average_scoring_time']
            )
    
    def _update_score_statistics(self, scores: List[RelevanceScore]):
        """Update score distribution statistics"""
        if scores:
            avg_score = sum(score.score for score in scores) / len(scores)
            
            if self.metrics['total_scoring_requests'] == 1:
                self.metrics['average_score'] = avg_score
            else:
                alpha = 0.1
                self.metrics['average_score'] = (
                    alpha * avg_score + 
                    (1 - alpha) * self.metrics['average_score']
                )
            
            # Update distribution
            for score in scores:
                if score.score >= 0.7:
                    self.metrics['score_distribution']['high'] += 1
                elif score.score >= 0.4:
                    self.metrics['score_distribution']['medium'] += 1
                else:
                    self.metrics['score_distribution']['low'] += 1
    
    async def record_user_feedback(self, query_id: str, document_id: str, 
                                 feedback_type: str, feedback_value: float,
                                 user_id: Optional[str] = None):
        """Record user feedback for learning"""
        try:
            feedback = {
                'query_id': query_id,
                'document_id': document_id,
                'feedback_type': feedback_type,
                'feedback_value': feedback_value,
                'timestamp': datetime.now().isoformat()
            }
            
            if user_id:
                if user_id not in self.user_feedback:
                    self.user_feedback[user_id] = []
                self.user_feedback[user_id].append(feedback)
                
                # Store in Redis
                if self.redis_client:
                    key = f"user_feedback:{user_id}"
                    data = json.dumps(self.user_feedback[user_id])
                    await self.redis_client.set(key, data)
            
            logger.info(f"✅ User feedback recorded: {feedback_type} = {feedback_value}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record user feedback: {e}")
    
    async def get_scoring_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scoring metrics"""
        return {
            'metrics': self.metrics.copy(),
            'corpus_size': self.total_documents,
            'vocabulary_size': len(self.document_frequencies),
            'scoring_history_size': len(self.scoring_history),
            'user_feedback_count': sum(len(feedback) for feedback in self.user_feedback.values()),
            'feature_importance': self.feature_importance.copy(),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global relevance scorer instance
relevance_scorer = RelevanceScorer()

async def initialize_relevance_scorer():
    """Initialize the global relevance scorer"""
    await relevance_scorer.initialize()

if __name__ == "__main__":
    # Test the relevance scorer
    async def test_relevance_scorer():
        await initialize_relevance_scorer()
        
        # Mock documents for testing
        mock_documents = [
            {
                'document_id': 'doc1',
                'content': 'Machine learning is a subset of artificial intelligence.',
                'document_metadata': {'document_type': 'article'}
            },
            {
                'document_id': 'doc2',
                'content': 'Deep learning uses neural networks with multiple layers.',
                'document_metadata': {'document_type': 'tutorial'}
            }
        ]
        
        # Test relevance scoring
        result = await relevance_scorer.score_relevance(
            "What is machine learning?",
            mock_documents,
            ScoringAlgorithm.HYBRID_SCORING
        )
        
        print(f"Scored {len(result.scores)} documents")
        for score in result.scores:
            print(f"- {score.document_id}: {score.score:.3f} ({score.algorithm.value})")
        
        # Get metrics
        metrics = await relevance_scorer.get_scoring_metrics()
        print(f"Scoring metrics: {metrics}")
    
    asyncio.run(test_relevance_scorer())
