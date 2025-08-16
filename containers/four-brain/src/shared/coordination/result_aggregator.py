"""
Result Aggregation System for Four-Brain System v2
Intelligent aggregation and synthesis of results from multiple brain instances

Created: 2025-07-30 AEST
Purpose: Combine and optimize results from Brain1, Brain2, Brain3, and Brain4
"""

import asyncio
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AggregationStrategy(Enum):
    """Result aggregation strategies"""
    SIMPLE_MERGE = "simple_merge"
    WEIGHTED_AVERAGE = "weighted_average"
    CONSENSUS_VOTING = "consensus_voting"
    BEST_RESULT = "best_result"
    ENSEMBLE_FUSION = "ensemble_fusion"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    TEMPORAL_FUSION = "temporal_fusion"

class ResultType(Enum):
    """Types of results that can be aggregated"""
    EMBEDDING = "embedding"
    RANKING = "ranking"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"

class QualityMetric(Enum):
    """Quality metrics for result evaluation"""
    CONFIDENCE = "confidence"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    LATENCY = "latency"
    RESOURCE_EFFICIENCY = "resource_efficiency"

@dataclass
class BrainResult:
    """Result from individual brain"""
    brain_id: str
    task_id: str
    result_type: ResultType
    data: Any
    confidence: float
    quality_metrics: Dict[str, float]
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class AggregationRequest:
    """Request for result aggregation"""
    request_id: str
    task_id: str
    result_type: ResultType
    strategy: AggregationStrategy
    brain_results: List[BrainResult]
    weights: Optional[Dict[str, float]]
    quality_requirements: Dict[str, float]
    timeout_seconds: float
    created_at: datetime

@dataclass
class AggregatedResult:
    """Final aggregated result"""
    request_id: str
    task_id: str
    result_type: ResultType
    strategy_used: AggregationStrategy
    aggregated_data: Any
    confidence: float
    quality_score: float
    contributing_brains: List[str]
    processing_time: float
    metadata: Dict[str, Any]
    timestamp: datetime

class ResultAggregator:
    """
    Intelligent result aggregation system
    
    Features:
    - Multiple aggregation strategies
    - Quality-aware result fusion
    - Confidence-weighted aggregation
    - Real-time result processing
    - Performance optimization
    - Adaptive strategy selection
    - Comprehensive quality assessment
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/17"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Aggregation state
        self.pending_requests: Dict[str, AggregationRequest] = {}
        self.completed_results: Dict[str, AggregatedResult] = {}
        self.brain_performance: Dict[str, Dict[str, float]] = {}
        
        # Configuration
        self.config = {
            'default_strategy': AggregationStrategy.CONFIDENCE_WEIGHTED,
            'min_results_for_aggregation': 2,
            'max_wait_time_seconds': 30,
            'quality_threshold': 0.7,
            'confidence_threshold': 0.5,
            'adaptive_strategy_enabled': True,
            'performance_tracking_enabled': True
        }
        
        # Strategy configurations
        self.strategy_configs = {
            AggregationStrategy.WEIGHTED_AVERAGE: {
                'default_weights': {'brain1': 0.3, 'brain2': 0.25, 'brain3': 0.25, 'brain4': 0.2},
                'normalize_weights': True
            },
            AggregationStrategy.CONSENSUS_VOTING: {
                'min_consensus_ratio': 0.6,
                'tie_breaking_strategy': 'confidence'
            },
            AggregationStrategy.CONFIDENCE_WEIGHTED: {
                'confidence_power': 2.0,
                'min_confidence': 0.1
            }
        }
        
        # Quality assessors
        self.quality_assessors: Dict[ResultType, Callable] = {}
        
        # Aggregation metrics
        self.metrics = {
            'total_requests': 0,
            'successful_aggregations': 0,
            'failed_aggregations': 0,
            'average_processing_time': 0.0,
            'average_quality_score': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in AggregationStrategy}
        }
        
        logger.info("🔗 Result Aggregator initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start aggregation services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize quality assessors
            self._initialize_quality_assessors()
            
            # Load brain performance history
            await self._load_brain_performance()
            
            # Start background services
            asyncio.create_task(self._aggregation_processor())
            asyncio.create_task(self._timeout_monitor())
            asyncio.create_task(self._performance_tracker())
            
            logger.info("✅ Result Aggregator Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Result Aggregator: {e}")
            raise
    
    async def submit_aggregation_request(self, task_id: str, result_type: ResultType,
                                       strategy: Optional[AggregationStrategy] = None,
                                       weights: Optional[Dict[str, float]] = None,
                                       quality_requirements: Optional[Dict[str, float]] = None,
                                       timeout_seconds: float = 30.0) -> str:
        """Submit request for result aggregation"""
        try:
            # Generate request ID
            request_id = f"agg_{int(time.time() * 1000)}_{len(self.pending_requests)}"
            
            # Use default strategy if not specified
            strategy = strategy or self.config['default_strategy']
            
            # Create aggregation request
            request = AggregationRequest(
                request_id=request_id,
                task_id=task_id,
                result_type=result_type,
                strategy=strategy,
                brain_results=[],
                weights=weights,
                quality_requirements=quality_requirements or {},
                timeout_seconds=timeout_seconds,
                created_at=datetime.now()
            )
            
            # Store request
            self.pending_requests[request_id] = request
            
            # Update metrics
            self.metrics['total_requests'] += 1
            
            logger.info(f"✅ Aggregation request submitted: {request_id} for task {task_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"❌ Failed to submit aggregation request: {e}")
            raise
    
    async def add_brain_result(self, request_id: str, brain_result: BrainResult) -> bool:
        """Add brain result to aggregation request"""
        try:
            request = self.pending_requests.get(request_id)
            if not request:
                logger.warning(f"Aggregation request not found: {request_id}")
                return False
            
            # Validate result
            if not await self._validate_brain_result(brain_result, request):
                logger.warning(f"Invalid brain result from {brain_result.brain_id}")
                return False
            
            # Add result
            request.brain_results.append(brain_result)
            
            # Check if ready for aggregation
            if len(request.brain_results) >= self.config['min_results_for_aggregation']:
                await self._trigger_aggregation(request)
            
            logger.debug(f"✅ Brain result added: {brain_result.brain_id} -> {request_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add brain result: {e}")
            return False
    
    async def _validate_brain_result(self, brain_result: BrainResult, request: AggregationRequest) -> bool:
        """Validate brain result against request requirements"""
        try:
            # Check result type match
            if brain_result.result_type != request.result_type:
                return False
            
            # Check task ID match
            if brain_result.task_id != request.task_id:
                return False
            
            # Check confidence threshold
            if brain_result.confidence < self.config['confidence_threshold']:
                return False
            
            # Check quality requirements
            for metric, min_value in request.quality_requirements.items():
                if metric in brain_result.quality_metrics:
                    if brain_result.quality_metrics[metric] < min_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Brain result validation failed: {e}")
            return False
    
    async def _trigger_aggregation(self, request: AggregationRequest):
        """Trigger aggregation process"""
        try:
            # Select optimal strategy if adaptive mode enabled
            if self.config['adaptive_strategy_enabled']:
                optimal_strategy = await self._select_optimal_strategy(request)
                if optimal_strategy != request.strategy:
                    request.strategy = optimal_strategy
                    logger.info(f"🔄 Strategy adapted to {optimal_strategy.value} for {request.request_id}")
            
            # Perform aggregation
            aggregated_result = await self._perform_aggregation(request)
            
            if aggregated_result:
                # Store result
                self.completed_results[request.request_id] = aggregated_result
                
                # Remove from pending
                self.pending_requests.pop(request.request_id, None)
                
                # Update metrics
                self.metrics['successful_aggregations'] += 1
                self.metrics['strategy_usage'][request.strategy.value] += 1
                self._update_average_processing_time(aggregated_result.processing_time)
                self._update_average_quality_score(aggregated_result.quality_score)
                
                # Store in Redis
                await self._store_aggregated_result(aggregated_result)
                
                logger.info(f"✅ Aggregation completed: {request.request_id}")
            else:
                self.metrics['failed_aggregations'] += 1
                logger.error(f"❌ Aggregation failed: {request.request_id}")
            
        except Exception as e:
            logger.error(f"❌ Aggregation trigger failed: {e}")
    
    async def _perform_aggregation(self, request: AggregationRequest) -> Optional[AggregatedResult]:
        """Perform result aggregation using specified strategy"""
        try:
            start_time = time.time()
            
            if request.strategy == AggregationStrategy.SIMPLE_MERGE:
                aggregated_data = await self._simple_merge(request.brain_results)
            elif request.strategy == AggregationStrategy.WEIGHTED_AVERAGE:
                aggregated_data = await self._weighted_average(request.brain_results, request.weights)
            elif request.strategy == AggregationStrategy.CONSENSUS_VOTING:
                aggregated_data = await self._consensus_voting(request.brain_results)
            elif request.strategy == AggregationStrategy.BEST_RESULT:
                aggregated_data = await self._best_result(request.brain_results)
            elif request.strategy == AggregationStrategy.CONFIDENCE_WEIGHTED:
                aggregated_data = await self._confidence_weighted(request.brain_results)
            elif request.strategy == AggregationStrategy.ENSEMBLE_FUSION:
                aggregated_data = await self._ensemble_fusion(request.brain_results)
            else:
                aggregated_data = await self._confidence_weighted(request.brain_results)  # Default
            
            if aggregated_data is None:
                return None
            
            # Calculate aggregated confidence
            confidences = [result.confidence for result in request.brain_results]
            aggregated_confidence = await self._calculate_aggregated_confidence(confidences, request.strategy)
            
            # Calculate quality score
            quality_score = await self._calculate_quality_score(request.brain_results, aggregated_data)
            
            # Create aggregated result
            processing_time = time.time() - start_time
            
            return AggregatedResult(
                request_id=request.request_id,
                task_id=request.task_id,
                result_type=request.result_type,
                strategy_used=request.strategy,
                aggregated_data=aggregated_data,
                confidence=aggregated_confidence,
                quality_score=quality_score,
                contributing_brains=[result.brain_id for result in request.brain_results],
                processing_time=processing_time,
                metadata={
                    'num_results': len(request.brain_results),
                    'strategy_config': self.strategy_configs.get(request.strategy, {})
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Aggregation performance failed: {e}")
            return None
    
    async def _simple_merge(self, brain_results: List[BrainResult]) -> Any:
        """Simple merge aggregation strategy"""
        try:
            if not brain_results:
                return None
            
            # For simple merge, just combine all data
            if brain_results[0].result_type == ResultType.EMBEDDING:
                # Average embeddings
                embeddings = [result.data for result in brain_results if isinstance(result.data, (list, np.ndarray))]
                if embeddings:
                    return np.mean(embeddings, axis=0).tolist()
            
            elif brain_results[0].result_type == ResultType.RANKING:
                # Merge rankings by averaging scores
                all_rankings = {}
                for result in brain_results:
                    if isinstance(result.data, dict):
                        for item, score in result.data.items():
                            if item not in all_rankings:
                                all_rankings[item] = []
                            all_rankings[item].append(score)
                
                # Average scores
                merged_rankings = {
                    item: statistics.mean(scores) 
                    for item, scores in all_rankings.items()
                }
                return dict(sorted(merged_rankings.items(), key=lambda x: x[1], reverse=True))
            
            else:
                # For other types, return the first result
                return brain_results[0].data
            
        except Exception as e:
            logger.error(f"❌ Simple merge failed: {e}")
            return None
    
    async def _weighted_average(self, brain_results: List[BrainResult], weights: Optional[Dict[str, float]]) -> Any:
        """Weighted average aggregation strategy"""
        try:
            if not brain_results:
                return None
            
            # Get weights
            if not weights:
                weights = self.strategy_configs[AggregationStrategy.WEIGHTED_AVERAGE]['default_weights']
            
            # Normalize weights if needed
            if self.strategy_configs[AggregationStrategy.WEIGHTED_AVERAGE]['normalize_weights']:
                total_weight = sum(weights.get(result.brain_id, 1.0) for result in brain_results)
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
            
            if brain_results[0].result_type == ResultType.EMBEDDING:
                # Weighted average of embeddings
                weighted_embedding = None
                total_weight = 0
                
                for result in brain_results:
                    weight = weights.get(result.brain_id, 1.0)
                    if isinstance(result.data, (list, np.ndarray)):
                        embedding = np.array(result.data)
                        if weighted_embedding is None:
                            weighted_embedding = embedding * weight
                        else:
                            weighted_embedding += embedding * weight
                        total_weight += weight
                
                if weighted_embedding is not None and total_weight > 0:
                    return (weighted_embedding / total_weight).tolist()
            
            elif brain_results[0].result_type == ResultType.RANKING:
                # Weighted ranking aggregation
                all_rankings = {}
                
                for result in brain_results:
                    weight = weights.get(result.brain_id, 1.0)
                    if isinstance(result.data, dict):
                        for item, score in result.data.items():
                            if item not in all_rankings:
                                all_rankings[item] = []
                            all_rankings[item].append(score * weight)
                
                # Calculate weighted averages
                weighted_rankings = {
                    item: sum(scores) / len(scores)
                    for item, scores in all_rankings.items()
                }
                return dict(sorted(weighted_rankings.items(), key=lambda x: x[1], reverse=True))
            
            return brain_results[0].data  # Fallback
            
        except Exception as e:
            logger.error(f"❌ Weighted average failed: {e}")
            return None
    
    async def _confidence_weighted(self, brain_results: List[BrainResult]) -> Any:
        """Confidence-weighted aggregation strategy"""
        try:
            if not brain_results:
                return None
            
            # Create weights based on confidence
            confidence_power = self.strategy_configs[AggregationStrategy.CONFIDENCE_WEIGHTED]['confidence_power']
            weights = {}
            
            for result in brain_results:
                confidence = max(result.confidence, self.strategy_configs[AggregationStrategy.CONFIDENCE_WEIGHTED]['min_confidence'])
                weights[result.brain_id] = confidence ** confidence_power
            
            # Use weighted average with confidence weights
            return await self._weighted_average(brain_results, weights)
            
        except Exception as e:
            logger.error(f"❌ Confidence weighted aggregation failed: {e}")
            return None
    
    async def _consensus_voting(self, brain_results: List[BrainResult]) -> Any:
        """Consensus voting aggregation strategy"""
        try:
            if not brain_results:
                return None
            
            min_consensus = self.strategy_configs[AggregationStrategy.CONSENSUS_VOTING]['min_consensus_ratio']
            
            if brain_results[0].result_type == ResultType.CLASSIFICATION:
                # Vote on classifications
                votes = {}
                for result in brain_results:
                    prediction = result.data
                    votes[prediction] = votes.get(prediction, 0) + 1
                
                # Check for consensus
                total_votes = len(brain_results)
                for prediction, count in votes.items():
                    if count / total_votes >= min_consensus:
                        return prediction
                
                # No consensus, use tie-breaking
                tie_breaking = self.strategy_configs[AggregationStrategy.CONSENSUS_VOTING]['tie_breaking_strategy']
                if tie_breaking == 'confidence':
                    # Return result with highest confidence
                    best_result = max(brain_results, key=lambda r: r.confidence)
                    return best_result.data
            
            # For other types, fall back to confidence weighted
            return await self._confidence_weighted(brain_results)
            
        except Exception as e:
            logger.error(f"❌ Consensus voting failed: {e}")
            return None
    
    async def _best_result(self, brain_results: List[BrainResult]) -> Any:
        """Best result aggregation strategy"""
        try:
            if not brain_results:
                return None
            
            # Select result with highest confidence
            best_result = max(brain_results, key=lambda r: r.confidence)
            return best_result.data
            
        except Exception as e:
            logger.error(f"❌ Best result selection failed: {e}")
            return None
    
    async def _ensemble_fusion(self, brain_results: List[BrainResult]) -> Any:
        """Advanced ensemble fusion strategy"""
        try:
            if not brain_results:
                return None
            
            # This would implement advanced ensemble methods
            # For now, use confidence-weighted as a sophisticated fallback
            return await self._confidence_weighted(brain_results)
            
        except Exception as e:
            logger.error(f"❌ Ensemble fusion failed: {e}")
            return None
    
    async def _calculate_aggregated_confidence(self, confidences: List[float], strategy: AggregationStrategy) -> float:
        """Calculate aggregated confidence score"""
        try:
            if not confidences:
                return 0.0
            
            if strategy == AggregationStrategy.BEST_RESULT:
                return max(confidences)
            elif strategy == AggregationStrategy.CONSENSUS_VOTING:
                # Confidence based on agreement
                return statistics.mean(confidences) * (1.0 - statistics.stdev(confidences) if len(confidences) > 1 else 1.0)
            else:
                # Weighted average of confidences
                return statistics.mean(confidences)
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.5
    
    async def _calculate_quality_score(self, brain_results: List[BrainResult], aggregated_data: Any) -> float:
        """Calculate quality score for aggregated result"""
        try:
            if not brain_results:
                return 0.0
            
            # Get quality assessor for result type
            result_type = brain_results[0].result_type
            assessor = self.quality_assessors.get(result_type)
            
            if assessor:
                return await assessor(brain_results, aggregated_data)
            else:
                # Default quality score based on confidence and consistency
                confidences = [result.confidence for result in brain_results]
                avg_confidence = statistics.mean(confidences)
                
                # Consistency score (lower std dev = higher consistency)
                if len(confidences) > 1:
                    consistency = 1.0 - min(1.0, statistics.stdev(confidences))
                else:
                    consistency = 1.0
                
                return (avg_confidence + consistency) / 2.0
            
        except Exception as e:
            logger.error(f"❌ Quality score calculation failed: {e}")
            return 0.5
    
    async def _select_optimal_strategy(self, request: AggregationRequest) -> AggregationStrategy:
        """Select optimal aggregation strategy based on context"""
        try:
            # Analyze brain results to determine best strategy
            num_results = len(request.brain_results)
            confidences = [result.confidence for result in request.brain_results]
            
            # If only one result, use best result
            if num_results == 1:
                return AggregationStrategy.BEST_RESULT
            
            # If high confidence variance, use confidence weighting
            if len(confidences) > 1 and statistics.stdev(confidences) > 0.2:
                return AggregationStrategy.CONFIDENCE_WEIGHTED
            
            # If classification task with multiple results, use consensus
            if request.result_type == ResultType.CLASSIFICATION and num_results >= 3:
                return AggregationStrategy.CONSENSUS_VOTING
            
            # Default to confidence weighted
            return AggregationStrategy.CONFIDENCE_WEIGHTED
            
        except Exception as e:
            logger.error(f"❌ Strategy selection failed: {e}")
            return request.strategy  # Fallback to original strategy
    
    def _initialize_quality_assessors(self):
        """Initialize quality assessment functions for different result types"""
        async def embedding_quality_assessor(brain_results: List[BrainResult], aggregated_data: Any) -> float:
            # Quality based on consistency of embeddings
            embeddings = [np.array(result.data) for result in brain_results if isinstance(result.data, (list, np.ndarray))]
            if len(embeddings) < 2:
                return 1.0
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    similarities.append(sim)
            
            return statistics.mean(similarities) if similarities else 0.5
        
        async def ranking_quality_assessor(brain_results: List[BrainResult], aggregated_data: Any) -> float:
            # Quality based on ranking agreement
            if len(brain_results) < 2:
                return 1.0
            
            # Calculate rank correlation between results
            # Simplified implementation
            return 0.8  # Placeholder
        
        self.quality_assessors = {
            ResultType.EMBEDDING: embedding_quality_assessor,
            ResultType.RANKING: ranking_quality_assessor
        }
    
    async def _aggregation_processor(self):
        """Background aggregation processor"""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                # Process pending requests that might be ready
                for request in list(self.pending_requests.values()):
                    if len(request.brain_results) >= self.config['min_results_for_aggregation']:
                        # Check if we should wait for more results
                        time_elapsed = (datetime.now() - request.created_at).total_seconds()
                        if time_elapsed >= request.timeout_seconds * 0.8:  # 80% of timeout
                            await self._trigger_aggregation(request)
                
            except Exception as e:
                logger.error(f"❌ Aggregation processor error: {e}")
    
    async def _timeout_monitor(self):
        """Monitor for request timeouts"""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                current_time = datetime.now()
                timed_out_requests = []
                
                for request in self.pending_requests.values():
                    time_elapsed = (current_time - request.created_at).total_seconds()
                    if time_elapsed >= request.timeout_seconds:
                        timed_out_requests.append(request)
                
                # Handle timed out requests
                for request in timed_out_requests:
                    if request.brain_results:
                        # Aggregate with available results
                        await self._trigger_aggregation(request)
                    else:
                        # No results available, mark as failed
                        self.pending_requests.pop(request.request_id, None)
                        self.metrics['failed_aggregations'] += 1
                        logger.warning(f"⏰ Aggregation request timed out: {request.request_id}")
                
            except Exception as e:
                logger.error(f"❌ Timeout monitor error: {e}")
    
    async def _performance_tracker(self):
        """Track brain performance for adaptive weighting"""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                # Analyze recent results to update brain performance
                await self._update_brain_performance()
                
            except Exception as e:
                logger.error(f"❌ Performance tracker error: {e}")
    
    async def _update_brain_performance(self):
        """Update brain performance metrics"""
        try:
            # Analyze recent aggregated results
            recent_results = [
                result for result in self.completed_results.values()
                if datetime.now() - result.timestamp < timedelta(hours=1)
            ]
            
            # Calculate performance metrics for each brain
            brain_metrics = {}
            for result in recent_results:
                for brain_id in result.contributing_brains:
                    if brain_id not in brain_metrics:
                        brain_metrics[brain_id] = {
                            'quality_scores': [],
                            'confidence_scores': [],
                            'processing_times': []
                        }
                    
                    brain_metrics[brain_id]['quality_scores'].append(result.quality_score)
                    brain_metrics[brain_id]['confidence_scores'].append(result.confidence)
                    brain_metrics[brain_id]['processing_times'].append(result.processing_time)
            
            # Update performance tracking
            for brain_id, metrics in brain_metrics.items():
                if brain_id not in self.brain_performance:
                    self.brain_performance[brain_id] = {}
                
                if metrics['quality_scores']:
                    self.brain_performance[brain_id]['avg_quality'] = statistics.mean(metrics['quality_scores'])
                if metrics['confidence_scores']:
                    self.brain_performance[brain_id]['avg_confidence'] = statistics.mean(metrics['confidence_scores'])
                if metrics['processing_times']:
                    self.brain_performance[brain_id]['avg_processing_time'] = statistics.mean(metrics['processing_times'])
            
        except Exception as e:
            logger.error(f"❌ Brain performance update failed: {e}")
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time metric"""
        if self.metrics['successful_aggregations'] == 1:
            self.metrics['average_processing_time'] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['average_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics['average_processing_time']
            )
    
    def _update_average_quality_score(self, quality_score: float):
        """Update average quality score metric"""
        if self.metrics['successful_aggregations'] == 1:
            self.metrics['average_quality_score'] = quality_score
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['average_quality_score'] = (
                alpha * quality_score + 
                (1 - alpha) * self.metrics['average_quality_score']
            )
    
    async def _store_aggregated_result(self, result: AggregatedResult):
        """Store aggregated result in Redis"""
        if self.redis_client:
            try:
                key = f"aggregated_result:{result.request_id}"
                data = json.dumps(asdict(result), default=str)
                await self.redis_client.setex(key, 86400, data)  # 24 hour TTL
            except Exception as e:
                logger.error(f"Failed to store aggregated result: {e}")
    
    async def _load_brain_performance(self):
        """Load brain performance history from Redis"""
        if self.redis_client:
            try:
                key = "brain_performance"
                data = await self.redis_client.get(key)
                if data:
                    self.brain_performance = json.loads(data)
            except Exception as e:
                logger.error(f"Failed to load brain performance: {e}")
    
    async def get_aggregated_result(self, request_id: str) -> Optional[AggregatedResult]:
        """Get aggregated result by request ID"""
        return self.completed_results.get(request_id)
    
    async def get_aggregation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive aggregation metrics"""
        return {
            'metrics': self.metrics.copy(),
            'pending_requests': len(self.pending_requests),
            'completed_results': len(self.completed_results),
            'brain_performance': self.brain_performance.copy(),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global result aggregator instance
result_aggregator = ResultAggregator()

async def initialize_result_aggregator():
    """Initialize the global result aggregator"""
    await result_aggregator.initialize()

if __name__ == "__main__":
    # Test the result aggregator
    async def test_result_aggregator():
        await initialize_result_aggregator()
        
        # Submit aggregation request
        request_id = await result_aggregator.submit_aggregation_request(
            "task123", ResultType.EMBEDDING, AggregationStrategy.CONFIDENCE_WEIGHTED
        )
        
        # Add brain results
        brain_result1 = BrainResult(
            brain_id="brain1",
            task_id="task123",
            result_type=ResultType.EMBEDDING,
            data=[0.1, 0.2, 0.3],
            confidence=0.9,
            quality_metrics={'accuracy': 0.85},
            processing_time=0.5,
            timestamp=datetime.now(),
            metadata={}
        )
        
        brain_result2 = BrainResult(
            brain_id="brain2",
            task_id="task123",
            result_type=ResultType.EMBEDDING,
            data=[0.15, 0.25, 0.35],
            confidence=0.8,
            quality_metrics={'accuracy': 0.80},
            processing_time=0.7,
            timestamp=datetime.now(),
            metadata={}
        )
        
        await result_aggregator.add_brain_result(request_id, brain_result1)
        await result_aggregator.add_brain_result(request_id, brain_result2)
        
        # Wait for aggregation
        await asyncio.sleep(2)
        
        # Get result
        result = await result_aggregator.get_aggregated_result(request_id)
        print(f"Aggregated result: {result}")
        
        # Get metrics
        metrics = await result_aggregator.get_aggregation_metrics()
        print(f"Aggregation metrics: {metrics}")
    
    asyncio.run(test_result_aggregator())
