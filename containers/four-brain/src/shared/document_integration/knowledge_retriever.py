"""
Knowledge Retriever for Four-Brain System v2
Intelligent retrieval of relevant document content for chat integration

Created: 2025-07-30 AEST
Purpose: Retrieve relevant document content based on user queries for Brain-3 chat
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Knowledge retrieval strategies"""
    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID_SEARCH = "hybrid_search"
    CONTEXTUAL_SEARCH = "contextual_search"
    TEMPORAL_SEARCH = "temporal_search"

class RelevanceType(Enum):
    """Types of relevance scoring"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    CONTEXTUAL = "contextual"
    TEMPORAL = "temporal"
    COMBINED = "combined"

@dataclass
class RetrievalQuery:
    """Knowledge retrieval query"""
    query_id: str
    query_text: str
    query_embedding: Optional[List[float]]
    strategy: RetrievalStrategy
    max_results: int
    min_relevance_score: float
    context_window: int
    filters: Dict[str, Any]
    user_id: Optional[str]
    session_id: Optional[str]
    created_at: datetime

@dataclass
class RetrievedChunk:
    """Retrieved document chunk with relevance scoring"""
    chunk_id: str
    document_id: str
    content: str
    relevance_score: float
    relevance_type: RelevanceType
    chunk_index: int
    document_title: str
    document_metadata: Dict[str, Any]
    chunk_metadata: Dict[str, Any]
    retrieval_context: Dict[str, Any]
    retrieved_at: datetime

@dataclass
class RetrievalResult:
    """Complete retrieval result"""
    query_id: str
    query_text: str
    strategy_used: RetrievalStrategy
    chunks: List[RetrievedChunk]
    total_chunks_found: int
    processing_time: float
    relevance_distribution: Dict[str, int]
    metadata: Dict[str, Any]
    retrieved_at: datetime

@dataclass
class RetrievalContext:
    """Context for retrieval operations"""
    user_id: Optional[str]
    session_id: Optional[str]
    conversation_history: List[str]
    current_topic: Optional[str]
    user_preferences: Dict[str, Any]
    temporal_context: Dict[str, Any]

class KnowledgeRetriever:
    """
    Intelligent knowledge retrieval system for document-based chat
    
    Features:
    - Multiple retrieval strategies (semantic, keyword, hybrid)
    - Advanced relevance scoring and ranking
    - Context-aware retrieval with conversation history
    - User preference learning and adaptation
    - Temporal relevance consideration
    - Performance optimization with caching
    - Real-time retrieval analytics
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/21"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Configuration
        self.config = {
            'default_max_results': 10,
            'min_relevance_threshold': 0.3,
            'semantic_weight': 0.6,
            'keyword_weight': 0.3,
            'contextual_weight': 0.1,
            'context_window_size': 3,  # chunks before/after
            'cache_ttl': 3600,  # seconds
            'embedding_dimensions': 2000,
            'max_query_length': 1000,
            'retrieval_timeout': 30  # seconds
        }
        
        # Retrieval state
        self.query_cache: Dict[str, RetrievalResult] = {}
        self.user_contexts: Dict[str, RetrievalContext] = {}
        self.retrieval_history: List[RetrievalResult] = []
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_retrieval_time': 0.0,
            'average_relevance_score': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in RetrievalStrategy},
            'user_satisfaction_score': 0.0
        }
        
        logger.info("🔍 Knowledge Retriever initialized")
    
    async def initialize(self):
        """Initialize Redis connection and retrieval services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load user contexts and preferences
            await self._load_user_contexts()
            
            # Start background services
            asyncio.create_task(self._cache_maintenance())
            asyncio.create_task(self._analytics_processor())
            
            logger.info("✅ Knowledge Retriever Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Knowledge Retriever: {e}")
            raise
    
    async def retrieve_knowledge(self, query_text: str, 
                                strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_SEARCH,
                                max_results: int = None,
                                min_relevance_score: float = None,
                                filters: Optional[Dict[str, Any]] = None,
                                user_id: Optional[str] = None,
                                session_id: Optional[str] = None) -> RetrievalResult:
        """Retrieve relevant knowledge for query"""
        try:
            start_time = time.time()
            
            # Generate query ID
            query_id = f"query_{int(time.time() * 1000)}_{len(self.retrieval_history)}"
            
            # Set defaults
            max_results = max_results or self.config['default_max_results']
            min_relevance_score = min_relevance_score or self.config['min_relevance_threshold']
            filters = filters or {}
            
            # Check cache first
            cache_key = self._generate_cache_key(query_text, strategy, max_results, filters)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                self.metrics['cache_hits'] += 1
                logger.debug(f"🎯 Cache hit for query: {query_text[:50]}...")
                return cached_result
            
            self.metrics['cache_misses'] += 1
            
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query_text)
            
            # Create retrieval query
            retrieval_query = RetrievalQuery(
                query_id=query_id,
                query_text=query_text,
                query_embedding=query_embedding,
                strategy=strategy,
                max_results=max_results,
                min_relevance_score=min_relevance_score,
                context_window=self.config['context_window_size'],
                filters=filters,
                user_id=user_id,
                session_id=session_id,
                created_at=datetime.now()
            )
            
            # Get user context
            user_context = await self._get_user_context(user_id, session_id)
            
            # Perform retrieval based on strategy
            retrieved_chunks = await self._execute_retrieval_strategy(retrieval_query, user_context)
            
            # Post-process and rank results
            final_chunks = await self._post_process_results(retrieved_chunks, retrieval_query, user_context)
            
            # Create retrieval result
            processing_time = time.time() - start_time
            
            result = RetrievalResult(
                query_id=query_id,
                query_text=query_text,
                strategy_used=strategy,
                chunks=final_chunks,
                total_chunks_found=len(retrieved_chunks),
                processing_time=processing_time,
                relevance_distribution=self._calculate_relevance_distribution(final_chunks),
                metadata={
                    'user_id': user_id,
                    'session_id': session_id,
                    'filters_applied': filters,
                    'context_used': bool(user_context)
                },
                retrieved_at=datetime.now()
            )
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            # Update metrics
            self.metrics['total_queries'] += 1
            self.metrics['strategy_usage'][strategy.value] += 1
            self._update_average_retrieval_time(processing_time)
            self._update_average_relevance_score(final_chunks)
            
            # Store in history
            self.retrieval_history.append(result)
            
            # Update user context
            await self._update_user_context(user_id, session_id, query_text, result)
            
            logger.info(f"✅ Knowledge retrieved: {len(final_chunks)} chunks for '{query_text[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"❌ Knowledge retrieval failed: {e}")
            raise
    
    async def _execute_retrieval_strategy(self, query: RetrievalQuery, 
                                        context: Optional[RetrievalContext]) -> List[RetrievedChunk]:
        """Execute specific retrieval strategy"""
        try:
            if query.strategy == RetrievalStrategy.SEMANTIC_SEARCH:
                return await self._semantic_search(query, context)
            elif query.strategy == RetrievalStrategy.KEYWORD_SEARCH:
                return await self._keyword_search(query, context)
            elif query.strategy == RetrievalStrategy.HYBRID_SEARCH:
                return await self._hybrid_search(query, context)
            elif query.strategy == RetrievalStrategy.CONTEXTUAL_SEARCH:
                return await self._contextual_search(query, context)
            elif query.strategy == RetrievalStrategy.TEMPORAL_SEARCH:
                return await self._temporal_search(query, context)
            else:
                # Default to hybrid search
                return await self._hybrid_search(query, context)
            
        except Exception as e:
            logger.error(f"❌ Retrieval strategy execution failed: {e}")
            return []
    
    async def _semantic_search(self, query: RetrievalQuery, 
                             context: Optional[RetrievalContext]) -> List[RetrievedChunk]:
        """Perform semantic search using embeddings"""
        try:
            retrieved_chunks = []
            
            if not query.query_embedding:
                return retrieved_chunks
            
            # Get all document chunks from Redis
            chunk_keys = await self.redis_client.keys("doc_chunk:*")
            
            for key in chunk_keys:
                chunk_data = await self.redis_client.get(key)
                if chunk_data:
                    chunk_dict = json.loads(chunk_data)
                    
                    # Calculate semantic similarity
                    if chunk_dict.get('embedding'):
                        chunk_embedding = chunk_dict['embedding']
                        similarity = self._calculate_cosine_similarity(
                            query.query_embedding, chunk_embedding
                        )
                        
                        if similarity >= query.min_relevance_score:
                            retrieved_chunk = RetrievedChunk(
                                chunk_id=chunk_dict['chunk_id'],
                                document_id=chunk_dict['document_id'],
                                content=chunk_dict['content'],
                                relevance_score=similarity,
                                relevance_type=RelevanceType.SEMANTIC,
                                chunk_index=chunk_dict['chunk_index'],
                                document_title=await self._get_document_title(chunk_dict['document_id']),
                                document_metadata=await self._get_document_metadata(chunk_dict['document_id']),
                                chunk_metadata=chunk_dict.get('metadata', {}),
                                retrieval_context={'strategy': 'semantic'},
                                retrieved_at=datetime.now()
                            )
                            retrieved_chunks.append(retrieved_chunk)
            
            # Sort by relevance score
            retrieved_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return retrieved_chunks[:query.max_results]
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    async def _keyword_search(self, query: RetrievalQuery, 
                            context: Optional[RetrievalContext]) -> List[RetrievedChunk]:
        """Perform keyword-based search"""
        try:
            retrieved_chunks = []
            query_keywords = query.query_text.lower().split()
            
            # Get all document chunks from Redis
            chunk_keys = await self.redis_client.keys("doc_chunk:*")
            
            for key in chunk_keys:
                chunk_data = await self.redis_client.get(key)
                if chunk_data:
                    chunk_dict = json.loads(chunk_data)
                    content = chunk_dict['content'].lower()
                    
                    # Calculate keyword relevance
                    keyword_matches = sum(1 for keyword in query_keywords if keyword in content)
                    relevance_score = keyword_matches / len(query_keywords) if query_keywords else 0
                    
                    if relevance_score >= query.min_relevance_score:
                        retrieved_chunk = RetrievedChunk(
                            chunk_id=chunk_dict['chunk_id'],
                            document_id=chunk_dict['document_id'],
                            content=chunk_dict['content'],
                            relevance_score=relevance_score,
                            relevance_type=RelevanceType.KEYWORD,
                            chunk_index=chunk_dict['chunk_index'],
                            document_title=await self._get_document_title(chunk_dict['document_id']),
                            document_metadata=await self._get_document_metadata(chunk_dict['document_id']),
                            chunk_metadata=chunk_dict.get('metadata', {}),
                            retrieval_context={'strategy': 'keyword', 'matches': keyword_matches},
                            retrieved_at=datetime.now()
                        )
                        retrieved_chunks.append(retrieved_chunk)
            
            # Sort by relevance score
            retrieved_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return retrieved_chunks[:query.max_results]
            
        except Exception as e:
            logger.error(f"❌ Keyword search failed: {e}")
            return []
    
    async def _hybrid_search(self, query: RetrievalQuery, 
                           context: Optional[RetrievalContext]) -> List[RetrievedChunk]:
        """Perform hybrid search combining semantic and keyword approaches"""
        try:
            # Get results from both strategies
            semantic_results = await self._semantic_search(query, context)
            keyword_results = await self._keyword_search(query, context)
            
            # Combine and re-rank results
            combined_chunks = {}
            
            # Add semantic results
            for chunk in semantic_results:
                chunk_id = chunk.chunk_id
                combined_chunks[chunk_id] = chunk
                chunk.relevance_score *= self.config['semantic_weight']
                chunk.relevance_type = RelevanceType.COMBINED
            
            # Add keyword results
            for chunk in keyword_results:
                chunk_id = chunk.chunk_id
                if chunk_id in combined_chunks:
                    # Combine scores
                    combined_chunks[chunk_id].relevance_score += (
                        chunk.relevance_score * self.config['keyword_weight']
                    )
                else:
                    chunk.relevance_score *= self.config['keyword_weight']
                    chunk.relevance_type = RelevanceType.COMBINED
                    combined_chunks[chunk_id] = chunk
            
            # Sort by combined relevance score
            final_chunks = list(combined_chunks.values())
            final_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return final_chunks[:query.max_results]
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return []
    
    async def _contextual_search(self, query: RetrievalQuery, 
                               context: Optional[RetrievalContext]) -> List[RetrievedChunk]:
        """Perform context-aware search"""
        try:
            # Start with hybrid search
            base_results = await self._hybrid_search(query, context)
            
            if not context:
                return base_results
            
            # Apply contextual boosting
            for chunk in base_results:
                contextual_boost = 0.0
                
                # Boost based on conversation history
                if context.conversation_history:
                    for prev_query in context.conversation_history[-3:]:  # Last 3 queries
                        if any(word in chunk.content.lower() for word in prev_query.lower().split()):
                            contextual_boost += 0.1
                
                # Boost based on current topic
                if context.current_topic and context.current_topic.lower() in chunk.content.lower():
                    contextual_boost += 0.2
                
                # Apply contextual boost
                chunk.relevance_score += contextual_boost * self.config['contextual_weight']
                chunk.relevance_type = RelevanceType.CONTEXTUAL
            
            # Re-sort with contextual scores
            base_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return base_results
            
        except Exception as e:
            logger.error(f"❌ Contextual search failed: {e}")
            return []
    
    async def _temporal_search(self, query: RetrievalQuery, 
                             context: Optional[RetrievalContext]) -> List[RetrievedChunk]:
        """Perform temporal-aware search"""
        try:
            # Start with hybrid search
            base_results = await self._hybrid_search(query, context)
            
            # Apply temporal relevance
            current_time = datetime.now()
            
            for chunk in base_results:
                # Get document creation/modification time
                doc_metadata = chunk.document_metadata
                doc_time = doc_metadata.get('modified_time')
                
                if doc_time:
                    if isinstance(doc_time, str):
                        doc_time = datetime.fromisoformat(doc_time)
                    
                    # Calculate temporal relevance (more recent = higher score)
                    time_diff = (current_time - doc_time).days
                    temporal_factor = max(0, 1 - (time_diff / 365))  # Decay over a year
                    
                    chunk.relevance_score *= (0.8 + 0.2 * temporal_factor)
                    chunk.relevance_type = RelevanceType.TEMPORAL
            
            # Re-sort with temporal scores
            base_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return base_results
            
        except Exception as e:
            logger.error(f"❌ Temporal search failed: {e}")
            return []
    
    async def _post_process_results(self, chunks: List[RetrievedChunk], 
                                  query: RetrievalQuery,
                                  context: Optional[RetrievalContext]) -> List[RetrievedChunk]:
        """Post-process retrieval results"""
        try:
            # Remove duplicates
            unique_chunks = {}
            for chunk in chunks:
                if chunk.chunk_id not in unique_chunks:
                    unique_chunks[chunk.chunk_id] = chunk
                elif chunk.relevance_score > unique_chunks[chunk.chunk_id].relevance_score:
                    unique_chunks[chunk.chunk_id] = chunk
            
            processed_chunks = list(unique_chunks.values())
            
            # Apply filters
            if query.filters:
                processed_chunks = await self._apply_filters(processed_chunks, query.filters)
            
            # Final relevance threshold
            processed_chunks = [
                chunk for chunk in processed_chunks 
                if chunk.relevance_score >= query.min_relevance_score
            ]
            
            # Sort by final relevance score
            processed_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return processed_chunks[:query.max_results]
            
        except Exception as e:
            logger.error(f"❌ Post-processing failed: {e}")
            return chunks
    
    async def _generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for query using Brain-1"""
        try:
            # Attempt to use real Brain-1 embedding service
            try:
                from ...brains.embedding_service.core.brain1_manager import Brain1Manager
                brain1_manager = Brain1Manager()

                # Generate real embedding
                embedding_result = await brain1_manager.generate_embedding(query_text)

                if not embedding_result or 'embedding' not in embedding_result:
                    raise ValueError("Brain-1 embedding generation failed")

                embedding = embedding_result['embedding']
                logger.debug(f"✅ Generated {len(embedding)}-dim embedding for query")
                return embedding

            except ImportError:
                logger.error("❌ Brain-1 service not available")
                raise ValueError("PROCESSING FAILED: Brain-1 embedding service not available")

            except Exception as brain1_error:
                logger.error(f"❌ Brain-1 embedding failed: {str(brain1_error)}")
                raise ValueError(f"PROCESSING FAILED: Brain-1 embedding error - {str(brain1_error)}")

        except Exception as e:
            logger.error(f"❌ Query embedding generation failed: {str(e)}")
            return []
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            similarity = cosine_similarity(vec1, vec2)[0][0]
            return float(similarity)
        except Exception:
            return 0.0
    
    async def _get_document_title(self, document_id: str) -> str:
        """Get document title from Redis"""
        try:
            key = f"indexed_doc:{document_id}"
            data = await self.redis_client.get(key)
            if data:
                doc_dict = json.loads(data)
                return doc_dict.get('title', 'Unknown Document')
            return 'Unknown Document'
        except Exception:
            return 'Unknown Document'
    
    async def _get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """Get document metadata from Redis"""
        try:
            key = f"indexed_doc:{document_id}"
            data = await self.redis_client.get(key)
            if data:
                doc_dict = json.loads(data)
                return doc_dict.get('metadata', {})
            return {}
        except Exception:
            return {}
    
    async def _get_user_context(self, user_id: Optional[str], 
                              session_id: Optional[str]) -> Optional[RetrievalContext]:
        """Get user context for personalized retrieval"""
        try:
            if not user_id:
                return None
            
            context_key = f"{user_id}:{session_id}" if session_id else user_id
            return self.user_contexts.get(context_key)
            
        except Exception as e:
            logger.error(f"❌ Failed to get user context: {e}")
            return None
    
    async def _update_user_context(self, user_id: Optional[str], session_id: Optional[str],
                                 query_text: str, result: RetrievalResult):
        """Update user context based on retrieval"""
        try:
            if not user_id:
                return
            
            context_key = f"{user_id}:{session_id}" if session_id else user_id
            
            if context_key not in self.user_contexts:
                self.user_contexts[context_key] = RetrievalContext(
                    user_id=user_id,
                    session_id=session_id,
                    conversation_history=[],
                    current_topic=None,
                    user_preferences={},
                    temporal_context={}
                )
            
            context = self.user_contexts[context_key]
            
            # Update conversation history
            context.conversation_history.append(query_text)
            if len(context.conversation_history) > 10:  # Keep last 10 queries
                context.conversation_history = context.conversation_history[-10:]
            
            # Update current topic (simple keyword extraction)
            if result.chunks:
                # Extract common keywords from top results
                all_content = " ".join([chunk.content for chunk in result.chunks[:3]])
                words = all_content.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 4:  # Only consider longer words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                if word_freq:
                    context.current_topic = max(word_freq, key=word_freq.get)
            
        except Exception as e:
            logger.error(f"❌ Failed to update user context: {e}")
    
    def _generate_cache_key(self, query_text: str, strategy: RetrievalStrategy,
                          max_results: int, filters: Dict[str, Any]) -> str:
        """Generate cache key for query"""
        import hashlib
        key_data = f"{query_text}:{strategy.value}:{max_results}:{json.dumps(filters, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[RetrievalResult]:
        """Get cached retrieval result"""
        try:
            if self.redis_client:
                data = await self.redis_client.get(f"retrieval_cache:{cache_key}")
                if data:
                    # Would deserialize RetrievalResult
                    return None  # Placeholder
            return None
        except Exception:
            return None
    
    async def _cache_result(self, cache_key: str, result: RetrievalResult):
        """Cache retrieval result"""
        try:
            if self.redis_client:
                key = f"retrieval_cache:{cache_key}"
                data = json.dumps(asdict(result), default=str)
                await self.redis_client.setex(key, self.config['cache_ttl'], data)
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
    
    async def _apply_filters(self, chunks: List[RetrievedChunk], 
                           filters: Dict[str, Any]) -> List[RetrievedChunk]:
        """Apply filters to retrieval results"""
        try:
            filtered_chunks = chunks
            
            # Document type filter
            if 'document_type' in filters:
                doc_type = filters['document_type']
                filtered_chunks = [
                    chunk for chunk in filtered_chunks
                    if chunk.document_metadata.get('document_type') == doc_type
                ]
            
            # Date range filter
            if 'date_range' in filters:
                date_range = filters['date_range']
                start_date = datetime.fromisoformat(date_range['start'])
                end_date = datetime.fromisoformat(date_range['end'])
                
                filtered_chunks = [
                    chunk for chunk in filtered_chunks
                    if start_date <= chunk.retrieved_at <= end_date
                ]
            
            # Tag filter
            if 'tags' in filters:
                required_tags = set(filters['tags'])
                filtered_chunks = [
                    chunk for chunk in filtered_chunks
                    if required_tags.intersection(set(chunk.document_metadata.get('tags', [])))
                ]
            
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"❌ Filter application failed: {e}")
            return chunks
    
    def _calculate_relevance_distribution(self, chunks: List[RetrievedChunk]) -> Dict[str, int]:
        """Calculate relevance score distribution"""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for chunk in chunks:
            if chunk.relevance_score >= 0.8:
                distribution['high'] += 1
            elif chunk.relevance_score >= 0.5:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def _update_average_retrieval_time(self, processing_time: float):
        """Update average retrieval time metric"""
        if self.metrics['total_queries'] == 1:
            self.metrics['average_retrieval_time'] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['average_retrieval_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics['average_retrieval_time']
            )
    
    def _update_average_relevance_score(self, chunks: List[RetrievedChunk]):
        """Update average relevance score metric"""
        if chunks:
            avg_score = sum(chunk.relevance_score for chunk in chunks) / len(chunks)
            
            if self.metrics['total_queries'] == 1:
                self.metrics['average_relevance_score'] = avg_score
            else:
                # Exponential moving average
                alpha = 0.1
                self.metrics['average_relevance_score'] = (
                    alpha * avg_score + 
                    (1 - alpha) * self.metrics['average_relevance_score']
                )
    
    async def _load_user_contexts(self):
        """Load user contexts from Redis"""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys("user_context:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        # Would deserialize RetrievalContext
                        pass
        except Exception as e:
            logger.error(f"Failed to load user contexts: {e}")
    
    async def _cache_maintenance(self):
        """Maintain retrieval cache"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up expired cache entries
                if self.redis_client:
                    keys = await self.redis_client.keys("retrieval_cache:*")
                    # Redis TTL handles expiration automatically
                
            except Exception as e:
                logger.error(f"❌ Cache maintenance error: {e}")
    
    async def _analytics_processor(self):
        """Process retrieval analytics"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze retrieval patterns
                # Update user preferences
                # Optimize retrieval strategies
                
            except Exception as e:
                logger.error(f"❌ Analytics processor error: {e}")
    
    async def get_retrieval_metrics(self) -> Dict[str, Any]:
        """Get comprehensive retrieval metrics"""
        return {
            'metrics': self.metrics.copy(),
            'active_user_contexts': len(self.user_contexts),
            'cached_results': len(self.query_cache),
            'retrieval_history_size': len(self.retrieval_history),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global knowledge retriever instance
knowledge_retriever = KnowledgeRetriever()

async def initialize_knowledge_retriever():
    """Initialize the global knowledge retriever"""
    await knowledge_retriever.initialize()

if __name__ == "__main__":
    # Test the knowledge retriever
    async def test_knowledge_retriever():
        await initialize_knowledge_retriever()
        
        # Test knowledge retrieval
        result = await knowledge_retriever.retrieve_knowledge(
            "What is machine learning?",
            strategy=RetrievalStrategy.HYBRID_SEARCH,
            max_results=5,
            user_id="test_user"
        )
        
        print(f"Retrieved {len(result.chunks)} chunks")
        for chunk in result.chunks:
            print(f"- {chunk.document_title}: {chunk.relevance_score:.3f}")
        
        # Get metrics
        metrics = await knowledge_retriever.get_retrieval_metrics()
        print(f"Retrieval metrics: {metrics}")
    
    asyncio.run(test_knowledge_retriever())
