"""
Context Builder for Four-Brain System v2
Intelligent context building from documents for Brain-3 chat integration

Created: 2025-07-30 AEST
Purpose: Build rich context from retrieved documents for enhanced chat conversations
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context that can be built"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    HISTORICAL = "historical"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"

class ContextPriority(Enum):
    """Context priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SUPPLEMENTARY = "supplementary"

class ContextFormat(Enum):
    """Context output formats"""
    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    BULLET_POINTS = "bullet_points"
    QA_PAIRS = "qa_pairs"
    SUMMARY = "summary"

@dataclass
class ContextElement:
    """Individual context element"""
    element_id: str
    content: str
    context_type: ContextType
    priority: ContextPriority
    source_document: str
    source_chunk: str
    relevance_score: float
    confidence_score: float
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class ContextSection:
    """Organized context section"""
    section_id: str
    title: str
    context_type: ContextType
    elements: List[ContextElement]
    summary: str
    key_points: List[str]
    relationships: List[str]
    metadata: Dict[str, Any]

@dataclass
class ChatContext:
    """Complete chat context for Brain-3"""
    context_id: str
    query: str
    sections: List[ContextSection]
    total_elements: int
    context_summary: str
    key_insights: List[str]
    suggested_questions: List[str]
    context_format: ContextFormat
    token_count: int
    processing_time: float
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class ContextBuildingRequest:
    """Request for context building"""
    request_id: str
    query: str
    retrieved_chunks: List[Any]  # RetrievedChunk objects
    context_format: ContextFormat
    max_context_length: int
    priority_filter: Optional[ContextPriority]
    include_metadata: bool
    user_preferences: Dict[str, Any]
    created_at: datetime

class ContextBuilder:
    """
    Intelligent context building system for document-based chat
    
    Features:
    - Multi-type context organization (factual, procedural, conceptual)
    - Intelligent context prioritization and filtering
    - Adaptive context formatting for different use cases
    - Context relationship mapping and cross-referencing
    - Token-aware context optimization for LLM consumption
    - Context quality assessment and validation
    - Real-time context building with performance optimization
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/22"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Configuration
        self.config = {
            'max_context_tokens': 4000,  # Maximum tokens for Brain-3
            'max_elements_per_section': 10,
            'min_relevance_threshold': 0.3,
            'context_overlap_threshold': 0.8,
            'summary_max_length': 200,
            'key_points_max_count': 5,
            'suggested_questions_count': 3,
            'processing_timeout': 30  # seconds
        }
        
        # Context building state
        self.built_contexts: Dict[str, ChatContext] = {}
        self.context_templates: Dict[ContextType, Dict[str, Any]] = {}
        self.context_cache: Dict[str, ChatContext] = {}
        
        # Performance metrics
        self.metrics = {
            'contexts_built': 0,
            'average_build_time': 0.0,
            'average_context_length': 0.0,
            'average_relevance_score': 0.0,
            'context_type_distribution': {ct.value: 0 for ct in ContextType},
            'format_usage': {cf.value: 0 for cf in ContextFormat},
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("🏗️ Context Builder initialized")
    
    async def initialize(self):
        """Initialize Redis connection and context building services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize context templates
            self._initialize_context_templates()
            
            # Load cached contexts
            await self._load_cached_contexts()
            
            # Start background services
            asyncio.create_task(self._context_cache_maintenance())
            
            logger.info("✅ Context Builder Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Context Builder: {e}")
            raise
    
    async def build_context(self, query: str, retrieved_chunks: List[Any],
                          context_format: ContextFormat = ContextFormat.STRUCTURED,
                          max_context_length: int = None,
                          priority_filter: Optional[ContextPriority] = None,
                          include_metadata: bool = True,
                          user_preferences: Optional[Dict[str, Any]] = None) -> ChatContext:
        """Build comprehensive context from retrieved chunks"""
        try:
            start_time = time.time()
            
            # Generate request ID
            request_id = f"ctx_{int(time.time() * 1000)}_{len(self.built_contexts)}"
            
            # Set defaults
            max_context_length = max_context_length or self.config['max_context_tokens']
            user_preferences = user_preferences or {}
            
            # Check cache first
            cache_key = self._generate_cache_key(query, retrieved_chunks, context_format)
            cached_context = await self._get_cached_context(cache_key)
            
            if cached_context:
                self.metrics['cache_hits'] += 1
                logger.debug(f"🎯 Cache hit for context: {query[:50]}...")
                return cached_context
            
            self.metrics['cache_misses'] += 1
            
            # Create context building request
            request = ContextBuildingRequest(
                request_id=request_id,
                query=query,
                retrieved_chunks=retrieved_chunks,
                context_format=context_format,
                max_context_length=max_context_length,
                priority_filter=priority_filter,
                include_metadata=include_metadata,
                user_preferences=user_preferences,
                created_at=datetime.now()
            )
            
            # Extract context elements from chunks
            context_elements = await self._extract_context_elements(request)
            
            # Organize elements into sections
            context_sections = await self._organize_context_sections(context_elements, request)
            
            # Generate context summary and insights
            context_summary = await self._generate_context_summary(context_sections, query)
            key_insights = await self._extract_key_insights(context_sections, query)
            suggested_questions = await self._generate_suggested_questions(context_sections, query)
            
            # Format context according to specified format
            formatted_sections = await self._format_context_sections(context_sections, context_format)
            
            # Calculate token count and optimize if needed
            token_count = await self._calculate_token_count(formatted_sections, context_summary)
            if token_count > max_context_length:
                formatted_sections = await self._optimize_context_length(
                    formatted_sections, max_context_length, context_summary
                )
                token_count = await self._calculate_token_count(formatted_sections, context_summary)
            
            # Create final chat context
            processing_time = time.time() - start_time
            
            chat_context = ChatContext(
                context_id=request_id,
                query=query,
                sections=formatted_sections,
                total_elements=len(context_elements),
                context_summary=context_summary,
                key_insights=key_insights,
                suggested_questions=suggested_questions,
                context_format=context_format,
                token_count=token_count,
                processing_time=processing_time,
                metadata={
                    'source_chunks': len(retrieved_chunks),
                    'user_preferences': user_preferences,
                    'optimization_applied': token_count != await self._calculate_token_count(context_sections, context_summary)
                },
                created_at=datetime.now()
            )
            
            # Cache context
            await self._cache_context(cache_key, chat_context)
            
            # Store context
            self.built_contexts[request_id] = chat_context
            
            # Update metrics
            self.metrics['contexts_built'] += 1
            self.metrics['format_usage'][context_format.value] += 1
            self._update_average_build_time(processing_time)
            self._update_average_context_length(token_count)
            self._update_context_type_distribution(formatted_sections)
            
            logger.info(f"✅ Context built: {len(formatted_sections)} sections, {token_count} tokens for '{query[:50]}...'")
            return chat_context
            
        except Exception as e:
            logger.error(f"❌ Context building failed: {e}")
            raise
    
    async def _extract_context_elements(self, request: ContextBuildingRequest) -> List[ContextElement]:
        """Extract context elements from retrieved chunks"""
        try:
            context_elements = []
            
            for i, chunk in enumerate(request.retrieved_chunks):
                # Determine context type based on content analysis
                context_type = await self._classify_context_type(chunk.content)
                
                # Determine priority based on relevance and position
                priority = await self._determine_context_priority(chunk, i, len(request.retrieved_chunks))
                
                # Calculate confidence score
                confidence_score = await self._calculate_confidence_score(chunk, request.query)
                
                # Create context element
                element = ContextElement(
                    element_id=f"{request.request_id}_elem_{i}",
                    content=chunk.content,
                    context_type=context_type,
                    priority=priority,
                    source_document=chunk.document_title,
                    source_chunk=chunk.chunk_id,
                    relevance_score=chunk.relevance_score,
                    confidence_score=confidence_score,
                    metadata={
                        'chunk_index': chunk.chunk_index,
                        'document_metadata': chunk.document_metadata,
                        'retrieval_context': chunk.retrieval_context
                    },
                    created_at=datetime.now()
                )
                
                context_elements.append(element)
            
            # Filter by priority if specified
            if request.priority_filter:
                priority_order = [ContextPriority.CRITICAL, ContextPriority.HIGH, 
                                ContextPriority.MEDIUM, ContextPriority.LOW, ContextPriority.SUPPLEMENTARY]
                filter_index = priority_order.index(request.priority_filter)
                allowed_priorities = priority_order[:filter_index + 1]
                context_elements = [elem for elem in context_elements if elem.priority in allowed_priorities]
            
            # Sort by priority and relevance
            context_elements.sort(key=lambda x: (
                [ContextPriority.CRITICAL, ContextPriority.HIGH, ContextPriority.MEDIUM, 
                 ContextPriority.LOW, ContextPriority.SUPPLEMENTARY].index(x.priority),
                -x.relevance_score
            ))
            
            return context_elements
            
        except Exception as e:
            logger.error(f"❌ Context element extraction failed: {e}")
            return []
    
    async def _classify_context_type(self, content: str) -> ContextType:
        """Classify content into context type"""
        try:
            content_lower = content.lower()
            
            # Simple rule-based classification (would use ML in production)
            if any(word in content_lower for word in ['how to', 'step', 'procedure', 'process', 'method']):
                return ContextType.PROCEDURAL
            elif any(word in content_lower for word in ['definition', 'concept', 'theory', 'principle']):
                return ContextType.CONCEPTUAL
            elif any(word in content_lower for word in ['history', 'timeline', 'evolution', 'development']):
                return ContextType.HISTORICAL
            elif any(word in content_lower for word in ['compare', 'versus', 'difference', 'similarity']):
                return ContextType.COMPARATIVE
            elif any(word in content_lower for word in ['analysis', 'evaluation', 'assessment', 'conclusion']):
                return ContextType.ANALYTICAL
            else:
                return ContextType.FACTUAL
            
        except Exception as e:
            logger.error(f"❌ Context type classification failed: {e}")
            return ContextType.FACTUAL
    
    async def _determine_context_priority(self, chunk: Any, position: int, total_chunks: int) -> ContextPriority:
        """Determine context priority based on relevance and position"""
        try:
            relevance = chunk.relevance_score
            position_factor = 1 - (position / total_chunks)  # Earlier chunks get higher priority
            
            combined_score = relevance * 0.7 + position_factor * 0.3
            
            if combined_score >= 0.9:
                return ContextPriority.CRITICAL
            elif combined_score >= 0.7:
                return ContextPriority.HIGH
            elif combined_score >= 0.5:
                return ContextPriority.MEDIUM
            elif combined_score >= 0.3:
                return ContextPriority.LOW
            else:
                return ContextPriority.SUPPLEMENTARY
            
        except Exception as e:
            logger.error(f"❌ Priority determination failed: {e}")
            return ContextPriority.MEDIUM
    
    async def _calculate_confidence_score(self, chunk: Any, query: str) -> float:
        """Calculate confidence score for context element"""
        try:
            # Base confidence from relevance score
            base_confidence = chunk.relevance_score
            
            # Boost confidence based on query keyword matches
            query_words = set(query.lower().split())
            content_words = set(chunk.content.lower().split())
            keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
            
            # Boost confidence based on document metadata
            metadata_boost = 0.0
            if hasattr(chunk, 'document_metadata'):
                # Boost for recent documents
                if 'modified_time' in chunk.document_metadata:
                    # Would calculate recency boost
                    metadata_boost += 0.1
            
            # Combine factors
            confidence = min(1.0, base_confidence + keyword_overlap * 0.2 + metadata_boost)
            
            return confidence
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.5
    
    async def _organize_context_sections(self, elements: List[ContextElement], 
                                       request: ContextBuildingRequest) -> List[ContextSection]:
        """Organize context elements into logical sections"""
        try:
            sections = []
            
            # Group elements by context type
            type_groups = {}
            for element in elements:
                if element.context_type not in type_groups:
                    type_groups[element.context_type] = []
                type_groups[element.context_type].append(element)
            
            # Create sections for each type
            for context_type, type_elements in type_groups.items():
                if not type_elements:
                    continue
                
                # Limit elements per section
                limited_elements = type_elements[:self.config['max_elements_per_section']]
                
                # Generate section summary
                section_summary = await self._generate_section_summary(limited_elements, context_type)
                
                # Extract key points
                key_points = await self._extract_section_key_points(limited_elements)
                
                # Find relationships
                relationships = await self._find_element_relationships(limited_elements)
                
                section = ContextSection(
                    section_id=f"{request.request_id}_sec_{context_type.value}",
                    title=self._get_section_title(context_type),
                    context_type=context_type,
                    elements=limited_elements,
                    summary=section_summary,
                    key_points=key_points,
                    relationships=relationships,
                    metadata={
                        'element_count': len(limited_elements),
                        'avg_relevance': sum(e.relevance_score for e in limited_elements) / len(limited_elements),
                        'avg_confidence': sum(e.confidence_score for e in limited_elements) / len(limited_elements)
                    }
                )
                
                sections.append(section)
            
            # Sort sections by priority (based on highest priority element in each section)
            sections.sort(key=lambda s: min([
                [ContextPriority.CRITICAL, ContextPriority.HIGH, ContextPriority.MEDIUM, 
                 ContextPriority.LOW, ContextPriority.SUPPLEMENTARY].index(e.priority)
                for e in s.elements
            ]))
            
            return sections
            
        except Exception as e:
            logger.error(f"❌ Context section organization failed: {e}")
            return []
    
    async def _generate_section_summary(self, elements: List[ContextElement], 
                                      context_type: ContextType) -> str:
        """Generate summary for context section"""
        try:
            if not elements:
                return ""
            
            # Simple extractive summarization (would use advanced NLP in production)
            all_content = " ".join([elem.content for elem in elements])
            
            # Take first few sentences as summary
            sentences = all_content.split('. ')
            summary_sentences = sentences[:3]  # First 3 sentences
            summary = '. '.join(summary_sentences)
            
            # Truncate if too long
            if len(summary) > self.config['summary_max_length']:
                summary = summary[:self.config['summary_max_length']] + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Section summary generation failed: {e}")
            return "Summary not available"
    
    async def _extract_section_key_points(self, elements: List[ContextElement]) -> List[str]:
        """Extract key points from section elements"""
        try:
            key_points = []
            
            # Simple key point extraction (would use advanced NLP in production)
            for element in elements[:self.config['key_points_max_count']]:
                # Extract first sentence as key point
                sentences = element.content.split('. ')
                if sentences:
                    key_point = sentences[0].strip()
                    if key_point and len(key_point) > 10:  # Minimum length
                        key_points.append(key_point)
            
            return key_points[:self.config['key_points_max_count']]
            
        except Exception as e:
            logger.error(f"❌ Key point extraction failed: {e}")
            return []
    
    async def _find_element_relationships(self, elements: List[ContextElement]) -> List[str]:
        """Find relationships between context elements"""
        try:
            relationships = []
            
            # Simple relationship detection (would use advanced NLP in production)
            for i, elem1 in enumerate(elements):
                for j, elem2 in enumerate(elements[i+1:], i+1):
                    # Check for common keywords
                    words1 = set(elem1.content.lower().split())
                    words2 = set(elem2.content.lower().split())
                    
                    overlap = len(words1.intersection(words2))
                    if overlap > 5:  # Threshold for relationship
                        relationship = f"Related to {elem2.source_document}"
                        if relationship not in relationships:
                            relationships.append(relationship)
            
            return relationships[:3]  # Limit relationships
            
        except Exception as e:
            logger.error(f"❌ Relationship finding failed: {e}")
            return []
    
    def _get_section_title(self, context_type: ContextType) -> str:
        """Get human-readable title for context type"""
        titles = {
            ContextType.FACTUAL: "Key Facts",
            ContextType.PROCEDURAL: "Procedures & Methods",
            ContextType.CONCEPTUAL: "Concepts & Definitions",
            ContextType.HISTORICAL: "Historical Context",
            ContextType.COMPARATIVE: "Comparisons & Analysis",
            ContextType.ANALYTICAL: "Analysis & Insights"
        }
        return titles.get(context_type, "Additional Information")
    
    async def _generate_context_summary(self, sections: List[ContextSection], query: str) -> str:
        """Generate overall context summary"""
        try:
            if not sections:
                return "No relevant context found."
            
            # Combine section summaries
            section_summaries = [section.summary for section in sections if section.summary]
            
            if not section_summaries:
                return "Context available but summary could not be generated."
            
            # Create overall summary
            summary = f"Based on {len(sections)} sections of information, "
            summary += " ".join(section_summaries[:2])  # First 2 section summaries
            
            # Truncate if too long
            if len(summary) > self.config['summary_max_length'] * 2:
                summary = summary[:self.config['summary_max_length'] * 2] + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Context summary generation failed: {e}")
            return "Summary generation failed."
    
    async def _extract_key_insights(self, sections: List[ContextSection], query: str) -> List[str]:
        """Extract key insights from context sections"""
        try:
            insights = []
            
            # Extract insights from each section
            for section in sections:
                if section.key_points:
                    # Take the most relevant key point
                    insight = section.key_points[0]
                    insights.append(f"{section.title}: {insight}")
            
            return insights[:3]  # Limit to 3 insights
            
        except Exception as e:
            logger.error(f"❌ Key insight extraction failed: {e}")
            return []
    
    async def _generate_suggested_questions(self, sections: List[ContextSection], query: str) -> List[str]:
        """Generate suggested follow-up questions"""
        try:
            questions = []
            
            # Generate questions based on context types
            for section in sections:
                if section.context_type == ContextType.PROCEDURAL:
                    questions.append("What are the detailed steps for this process?")
                elif section.context_type == ContextType.CONCEPTUAL:
                    questions.append("Can you explain this concept in more detail?")
                elif section.context_type == ContextType.COMPARATIVE:
                    questions.append("What are the key differences between these options?")
                elif section.context_type == ContextType.ANALYTICAL:
                    questions.append("What are the implications of this analysis?")
            
            # Remove duplicates and limit
            unique_questions = list(dict.fromkeys(questions))
            return unique_questions[:self.config['suggested_questions_count']]
            
        except Exception as e:
            logger.error(f"❌ Suggested question generation failed: {e}")
            return []
    
    async def _format_context_sections(self, sections: List[ContextSection], 
                                     format_type: ContextFormat) -> List[ContextSection]:
        """Format context sections according to specified format"""
        try:
            if format_type == ContextFormat.STRUCTURED:
                return sections  # Already structured
            
            elif format_type == ContextFormat.NARRATIVE:
                return await self._format_as_narrative(sections)
            
            elif format_type == ContextFormat.BULLET_POINTS:
                return await self._format_as_bullet_points(sections)
            
            elif format_type == ContextFormat.QA_PAIRS:
                return await self._format_as_qa_pairs(sections)
            
            elif format_type == ContextFormat.SUMMARY:
                return await self._format_as_summary(sections)
            
            else:
                return sections  # Default to structured
            
        except Exception as e:
            logger.error(f"❌ Context formatting failed: {e}")
            return sections
    
    async def _format_as_narrative(self, sections: List[ContextSection]) -> List[ContextSection]:
        """Format sections as narrative text"""
        # Implementation would convert structured sections to narrative format
        return sections
    
    async def _format_as_bullet_points(self, sections: List[ContextSection]) -> List[ContextSection]:
        """Format sections as bullet points"""
        # Implementation would convert sections to bullet point format
        return sections
    
    async def _format_as_qa_pairs(self, sections: List[ContextSection]) -> List[ContextSection]:
        """Format sections as Q&A pairs"""
        # Implementation would convert sections to Q&A format
        return sections
    
    async def _format_as_summary(self, sections: List[ContextSection]) -> List[ContextSection]:
        """Format sections as summary"""
        # Implementation would create condensed summary format
        return sections
    
    async def _calculate_token_count(self, sections: List[ContextSection], summary: str) -> int:
        """Calculate approximate token count for context"""
        try:
            total_text = summary
            
            for section in sections:
                total_text += section.title + section.summary
                for element in section.elements:
                    total_text += element.content
                total_text += " ".join(section.key_points)
            
            # Rough token estimation (1 token ≈ 4 characters)
            return len(total_text) // 4
            
        except Exception as e:
            logger.error(f"❌ Token count calculation failed: {e}")
            return 0
    
    async def _optimize_context_length(self, sections: List[ContextSection], 
                                     max_tokens: int, summary: str) -> List[ContextSection]:
        """Optimize context length to fit within token limit"""
        try:
            # Start with highest priority sections
            optimized_sections = []
            current_tokens = len(summary) // 4
            
            for section in sections:
                section_tokens = await self._calculate_section_tokens(section)
                
                if current_tokens + section_tokens <= max_tokens:
                    optimized_sections.append(section)
                    current_tokens += section_tokens
                else:
                    # Try to include partial section
                    remaining_tokens = max_tokens - current_tokens
                    if remaining_tokens > 100:  # Minimum viable section size
                        truncated_section = await self._truncate_section(section, remaining_tokens)
                        if truncated_section:
                            optimized_sections.append(truncated_section)
                    break
            
            return optimized_sections
            
        except Exception as e:
            logger.error(f"❌ Context optimization failed: {e}")
            return sections
    
    async def _calculate_section_tokens(self, section: ContextSection) -> int:
        """Calculate token count for a section"""
        try:
            section_text = section.title + section.summary
            for element in section.elements:
                section_text += element.content
            section_text += " ".join(section.key_points)
            
            return len(section_text) // 4
            
        except Exception as e:
            logger.error(f"❌ Section token calculation failed: {e}")
            return 0
    
    async def _truncate_section(self, section: ContextSection, max_tokens: int) -> Optional[ContextSection]:
        """Truncate section to fit within token limit"""
        try:
            # Keep highest priority elements
            truncated_elements = []
            current_tokens = len(section.title + section.summary) // 4
            
            for element in section.elements:
                element_tokens = len(element.content) // 4
                if current_tokens + element_tokens <= max_tokens:
                    truncated_elements.append(element)
                    current_tokens += element_tokens
                else:
                    break
            
            if truncated_elements:
                section.elements = truncated_elements
                return section
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Section truncation failed: {e}")
            return None
    
    def _initialize_context_templates(self):
        """Initialize context templates for different types"""
        self.context_templates = {
            ContextType.FACTUAL: {
                'title_template': "Key Facts about {topic}",
                'summary_template': "The following facts are relevant to {query}:",
                'element_template': "• {content}"
            },
            ContextType.PROCEDURAL: {
                'title_template': "How to {action}",
                'summary_template': "Here are the procedures for {query}:",
                'element_template': "{step}. {content}"
            },
            # Add more templates as needed
        }
    
    def _generate_cache_key(self, query: str, chunks: List[Any], format_type: ContextFormat) -> str:
        """Generate cache key for context"""
        import hashlib
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        key_data = f"{query}:{format_type.value}:{':'.join(sorted(chunk_ids))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_cached_context(self, cache_key: str) -> Optional[ChatContext]:
        """Get cached context"""
        try:
            if self.redis_client:
                data = await self.redis_client.get(f"context_cache:{cache_key}")
                if data:
                    # Would deserialize ChatContext
                    return None  # Placeholder
            return None
        except Exception:
            return None
    
    async def _cache_context(self, cache_key: str, context: ChatContext):
        """Cache built context"""
        try:
            if self.redis_client:
                key = f"context_cache:{cache_key}"
                data = json.dumps(asdict(context), default=str)
                await self.redis_client.setex(key, 3600, data)  # 1 hour TTL
        except Exception as e:
            logger.error(f"Failed to cache context: {e}")
    
    async def _load_cached_contexts(self):
        """Load cached contexts from Redis"""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys("context_cache:*")
                # Would load cached contexts
                pass
        except Exception as e:
            logger.error(f"Failed to load cached contexts: {e}")
    
    async def _context_cache_maintenance(self):
        """Maintain context cache"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up expired contexts
                # Redis TTL handles expiration automatically
                
            except Exception as e:
                logger.error(f"❌ Context cache maintenance error: {e}")
    
    def _update_average_build_time(self, build_time: float):
        """Update average build time metric"""
        if self.metrics['contexts_built'] == 1:
            self.metrics['average_build_time'] = build_time
        else:
            alpha = 0.1
            self.metrics['average_build_time'] = (
                alpha * build_time + 
                (1 - alpha) * self.metrics['average_build_time']
            )
    
    def _update_average_context_length(self, token_count: int):
        """Update average context length metric"""
        if self.metrics['contexts_built'] == 1:
            self.metrics['average_context_length'] = token_count
        else:
            alpha = 0.1
            self.metrics['average_context_length'] = (
                alpha * token_count + 
                (1 - alpha) * self.metrics['average_context_length']
            )
    
    def _update_context_type_distribution(self, sections: List[ContextSection]):
        """Update context type distribution metrics"""
        for section in sections:
            self.metrics['context_type_distribution'][section.context_type.value] += 1
    
    async def get_context_metrics(self) -> Dict[str, Any]:
        """Get comprehensive context building metrics"""
        return {
            'metrics': self.metrics.copy(),
            'built_contexts': len(self.built_contexts),
            'cached_contexts': len(self.context_cache),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global context builder instance
context_builder = ContextBuilder()

async def initialize_context_builder():
    """Initialize the global context builder"""
    await context_builder.initialize()

if __name__ == "__main__":
    # Test the context builder
    async def test_context_builder():
        await initialize_context_builder()
        
        # Mock retrieved chunks for testing
        mock_chunks = []  # Would contain RetrievedChunk objects
        
        # Test context building
        context = await context_builder.build_context(
            "What is machine learning?",
            mock_chunks,
            ContextFormat.STRUCTURED
        )
        
        print(f"Built context with {len(context.sections)} sections")
        print(f"Token count: {context.token_count}")
        print(f"Summary: {context.context_summary}")
        
        # Get metrics
        metrics = await context_builder.get_context_metrics()
        print(f"Context metrics: {metrics}")
    
    asyncio.run(test_context_builder())
