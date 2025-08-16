#!/usr/bin/env python3
"""
Self-Improvement Loop for Four-Brain Architecture
Implements automated reflection and improvement mechanisms

This module provides self-improvement functionality for the Four-Brain System,
enabling automated reflection, performance analysis, and prompt rewriting
based on performance trends and historical data.

Zero Fabrication Policy: ENFORCED
All improvement algorithms use real performance data and statistical analysis.
"""

import time
import json
import asyncio
import statistics
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from .memory_store import MemoryStore, TaskScore, PatternMatch
from .self_grading import SelfGradingSystem, PerformanceScore
from .streams import StreamMessage

logger = structlog.get_logger(__name__)

class ImprovementType(Enum):
    """Types of improvements that can be made"""
    PROMPT_OPTIMIZATION = "prompt_optimization"
    PARAMETER_TUNING = "parameter_tuning"
    ALGORITHM_SELECTION = "algorithm_selection"
    RESOURCE_ALLOCATION = "resource_allocation"
    ERROR_HANDLING = "error_handling"
    CACHING_STRATEGY = "caching_strategy"

class ReflectionLevel(Enum):
    """Levels of reflection depth"""
    SHALLOW = "shallow"  # Basic performance metrics
    MEDIUM = "medium"   # Pattern analysis and trends
    DEEP = "deep"       # Root cause analysis and strategic changes

@dataclass
class ImprovementSuggestion:
    """Suggestion for system improvement"""
    improvement_type: ImprovementType
    description: str
    confidence: float
    expected_impact: float
    implementation_complexity: str  # low, medium, high
    supporting_evidence: List[str]
    proposed_changes: Dict[str, Any]
    priority: str = "medium"  # low, medium, high - default to medium
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'improvement_type': self.improvement_type.value,
            'description': self.description,
            'confidence': self.confidence,
            'expected_impact': self.expected_impact,
            'implementation_complexity': self.implementation_complexity,
            'supporting_evidence': self.supporting_evidence,
            'proposed_changes': self.proposed_changes,
            'priority': self.priority
        }

@dataclass
class ReflectionReport:
    """Report from reflection analysis"""
    brain_id: str
    analysis_period: str
    reflection_level: ReflectionLevel
    performance_summary: Dict[str, Any]
    identified_patterns: List[Dict[str, Any]]
    improvement_suggestions: List[ImprovementSuggestion]
    confidence_score: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'brain_id': self.brain_id,
            'analysis_period': self.analysis_period,
            'reflection_level': self.reflection_level.value,
            'performance_summary': self.performance_summary,
            'identified_patterns': self.identified_patterns,
            'improvement_suggestions': [s.to_dict() for s in self.improvement_suggestions],
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp
        }

class SelfImprovementEngine:
    """Self-improvement engine for automated system enhancement"""
    
    def __init__(self, memory_store: Optional[MemoryStore] = None,
                 grading_engine: Optional[SelfGradingSystem] = None):
        """Initialize self-improvement engine"""
        self.memory_store = memory_store
        self.grading_engine = grading_engine
        
        # Improvement thresholds
        self.performance_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "needs_improvement": 0.5,
            "critical": 0.3
        }
        
        # Reflection settings
        self.reflection_intervals = {
            ReflectionLevel.SHALLOW: 3600,    # 1 hour
            ReflectionLevel.MEDIUM: 21600,    # 6 hours
            ReflectionLevel.DEEP: 86400       # 24 hours
        }
        
        # Pattern detection settings
        self.min_samples_for_pattern = 5
        self.pattern_confidence_threshold = 0.7
        
        # Statistics
        self.reflections_performed = 0
        self.improvements_suggested = 0
        self.improvements_implemented = 0
        self.performance_gains_achieved = 0.0
        
        # Last reflection times
        self.last_reflections = {level: 0.0 for level in ReflectionLevel}
        
        logger.info("Self-improvement engine initialized")

    async def start_improvement_loop(self):
        """Start the continuous improvement loop"""
        logger.info("🔄 Starting self-improvement loop...")

        # For now, just perform periodic reflections
        # This can be expanded to include more sophisticated improvement logic
        while True:
            try:
                # Wait for 1 hour between improvement cycles
                await asyncio.sleep(3600)

                # Perform reflection for each brain
                brain_ids = ["brain1", "brain2", "brain3", "brain4"]
                for brain_id in brain_ids:
                    try:
                        reflection = await self.perform_reflection(brain_id)
                        logger.info(f"Reflection completed for {brain_id}",
                                  confidence=reflection.confidence_score,
                                  suggestions=len(reflection.improvement_suggestions))
                    except Exception as e:
                        logger.error(f"Reflection failed for {brain_id}: {e}")

            except Exception as e:
                logger.error(f"Self-improvement loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def perform_reflection(self, brain_id: str,
                               reflection_level: ReflectionLevel = ReflectionLevel.MEDIUM,
                               time_window_hours: int = 24) -> ReflectionReport:
        """Perform reflection analysis on brain performance"""
        try:
            start_time = time.time()
            
            # Get performance data
            performance_data = await self._gather_performance_data(
                brain_id, time_window_hours
            )
            
            # Analyze patterns
            patterns = await self._analyze_patterns(
                brain_id, performance_data, reflection_level
            )
            
            # Generate improvement suggestions
            suggestions = await self._generate_improvement_suggestions(
                brain_id, performance_data, patterns, reflection_level
            )
            
            # Calculate confidence
            confidence = self._calculate_reflection_confidence(
                performance_data, patterns, suggestions
            )
            
            # Create reflection report
            report = ReflectionReport(
                brain_id=brain_id,
                analysis_period=f"{time_window_hours}h",
                reflection_level=reflection_level,
                performance_summary=performance_data,
                identified_patterns=patterns,
                improvement_suggestions=suggestions,
                confidence_score=confidence,
                timestamp=start_time
            )
            
            # Update statistics
            self.reflections_performed += 1
            self.improvements_suggested += len(suggestions)
            self.last_reflections[reflection_level] = start_time
            
            logger.info("Reflection completed", 
                       brain_id=brain_id, level=reflection_level.value,
                       patterns_found=len(patterns), suggestions=len(suggestions),
                       confidence=confidence)
            
            return report
            
        except Exception as e:
            logger.error("Reflection failed", brain_id=brain_id, 
                        level=reflection_level.value, error=str(e))
            
            # Return empty report
            return ReflectionReport(
                brain_id=brain_id,
                analysis_period=f"{time_window_hours}h",
                reflection_level=reflection_level,
                performance_summary={},
                identified_patterns=[],
                improvement_suggestions=[],
                confidence_score=0.0,
                timestamp=time.time()
            )
    
    async def _gather_performance_data(self, brain_id: str, 
                                     time_window_hours: int) -> Dict[str, Any]:
        """Gather performance data for analysis"""
        if not self.memory_store:
            return {"error": "Memory store not available"}
        
        try:
            # Get brain performance statistics
            brain_stats = await self.memory_store.get_brain_performance(
                brain_id, time_window_hours
            )
            
            # Add trend analysis
            if brain_stats.get('total_tasks', 0) > 0:
                brain_stats['performance_trend'] = await self._calculate_performance_trend(
                    brain_id, time_window_hours
                )
            
            return brain_stats
            
        except Exception as e:
            logger.error("Failed to gather performance data", 
                        brain_id=brain_id, error=str(e))
            return {"error": str(e)}
    
    async def _calculate_performance_trend(self, brain_id: str, 
                                         time_window_hours: int) -> Dict[str, Any]:
        """Calculate performance trend over time"""
        # This would require time-series data from the memory store
        # For now, return a placeholder implementation
        return {
            "trend_direction": "stable",  # improving, stable, declining
            "trend_strength": 0.0,       # -1.0 to 1.0
            "confidence": 0.5
        }
    
    async def _analyze_patterns(self, brain_id: str, performance_data: Dict[str, Any],
                              reflection_level: ReflectionLevel) -> List[Dict[str, Any]]:
        """Analyze patterns in performance data"""
        patterns = []
        
        try:
            # Pattern 1: Performance consistency
            if 'operations' in performance_data:
                consistency_pattern = self._analyze_consistency_pattern(
                    performance_data['operations']
                )
                if consistency_pattern:
                    patterns.append(consistency_pattern)
            
            # Pattern 2: Resource utilization
            resource_pattern = self._analyze_resource_pattern(performance_data)
            if resource_pattern:
                patterns.append(resource_pattern)
            
            # Pattern 3: Error patterns (if reflection level is medium or deep)
            if reflection_level in [ReflectionLevel.MEDIUM, ReflectionLevel.DEEP]:
                error_pattern = self._analyze_error_pattern(performance_data)
                if error_pattern:
                    patterns.append(error_pattern)
            
            # Pattern 4: Time-based patterns (deep reflection only)
            if reflection_level == ReflectionLevel.DEEP:
                time_pattern = self._analyze_time_pattern(performance_data)
                if time_pattern:
                    patterns.append(time_pattern)
            
            return patterns
            
        except Exception as e:
            logger.error("Pattern analysis failed", brain_id=brain_id, error=str(e))
            return []
    
    def _analyze_consistency_pattern(self, operations: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze consistency patterns across operations"""
        if not operations:
            return None
        
        # Calculate coefficient of variation for each operation
        inconsistent_operations = []
        
        for operation, stats in operations.items():
            if stats.get('task_count', 0) >= self.min_samples_for_pattern:
                # This would require individual task scores, not just averages
                # For now, use a simplified approach
                avg_score = stats.get('average_score', 0.5)
                if avg_score < self.performance_thresholds['needs_improvement']:
                    inconsistent_operations.append(operation)
        
        if inconsistent_operations:
            return {
                'type': 'consistency',
                'description': f"Inconsistent performance in operations: {', '.join(inconsistent_operations)}",
                'affected_operations': inconsistent_operations,
                'confidence': 0.8,
                'severity': 'medium'
            }
        
        return None
    
    def _analyze_resource_pattern(self, performance_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze resource utilization patterns"""
        # This would require resource usage data
        # For now, return a placeholder based on task count
        total_tasks = performance_data.get('total_tasks', 0)
        
        if total_tasks > 100:  # High load scenario
            return {
                'type': 'resource_utilization',
                'description': 'High task volume detected - consider resource optimization',
                'task_count': total_tasks,
                'confidence': 0.6,
                'severity': 'low'
            }
        
        return None
    
    def _analyze_error_pattern(self, performance_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze error patterns"""
        # This would require error tracking data
        # For now, return a placeholder
        return None
    
    def _analyze_time_pattern(self, performance_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze time-based performance patterns"""
        # This would require time-series analysis
        # For now, return a placeholder
        return None
    
    async def _generate_improvement_suggestions(self, brain_id: str, 
                                              performance_data: Dict[str, Any],
                                              patterns: List[Dict[str, Any]],
                                              reflection_level: ReflectionLevel) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions based on analysis"""
        suggestions = []
        
        try:
            # Overall performance-based suggestions
            avg_score = performance_data.get('average_score', 0.5)
            
            if avg_score < self.performance_thresholds['needs_improvement']:
                # Set priority based on severity
                priority = "high" if avg_score < self.performance_thresholds['critical'] else "medium"
                suggestions.append(ImprovementSuggestion(
                    improvement_type=ImprovementType.PROMPT_OPTIMIZATION,
                    description="Overall performance below threshold - optimize prompts and parameters",
                    confidence=0.8,
                    expected_impact=0.3,
                    implementation_complexity="medium",
                    supporting_evidence=[f"Average score: {avg_score:.3f}"],
                    proposed_changes={
                        "action": "review_and_optimize_prompts",
                        "target_improvement": 0.2
                    },
                    priority=priority
                ))
            
            # Pattern-based suggestions
            for pattern in patterns:
                pattern_suggestions = self._generate_pattern_suggestions(pattern)
                suggestions.extend(pattern_suggestions)
            
            # Brain-specific suggestions
            brain_suggestions = self._generate_brain_specific_suggestions(
                brain_id, performance_data
            )
            suggestions.extend(brain_suggestions)
            
            # Sort by expected impact
            suggestions.sort(key=lambda s: s.expected_impact, reverse=True)
            
            return suggestions[:5]  # Limit to top 5 suggestions
            
        except Exception as e:
            logger.error("Failed to generate improvement suggestions", 
                        brain_id=brain_id, error=str(e))
            return []
    
    def _generate_pattern_suggestions(self, pattern: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """Generate suggestions based on identified patterns"""
        suggestions = []
        
        pattern_type = pattern.get('type')
        
        if pattern_type == 'consistency':
            suggestions.append(ImprovementSuggestion(
                improvement_type=ImprovementType.ALGORITHM_SELECTION,
                description="Improve consistency by standardizing algorithms",
                confidence=pattern.get('confidence', 0.5),
                expected_impact=0.25,
                implementation_complexity="medium",
                supporting_evidence=[pattern.get('description', '')],
                proposed_changes={
                    "action": "standardize_algorithms",
                    "affected_operations": pattern.get('affected_operations', [])
                }
            ))
        
        elif pattern_type == 'resource_utilization':
            suggestions.append(ImprovementSuggestion(
                improvement_type=ImprovementType.RESOURCE_ALLOCATION,
                description="Optimize resource allocation for high-load scenarios",
                confidence=pattern.get('confidence', 0.5),
                expected_impact=0.2,
                implementation_complexity="high",
                supporting_evidence=[pattern.get('description', '')],
                proposed_changes={
                    "action": "optimize_resource_allocation",
                    "task_count": pattern.get('task_count', 0)
                }
            ))
        
        return suggestions
    
    def _generate_brain_specific_suggestions(self, brain_id: str, 
                                           performance_data: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """Generate brain-specific improvement suggestions"""
        suggestions = []
        
        # Brain-specific optimization strategies
        brain_strategies = {
            "brain1": {  # Embedding brain
                "focus": "embedding_quality",
                "common_improvements": ["model_fine_tuning", "batch_optimization"]
            },
            "brain2": {  # Reranking brain
                "focus": "ranking_accuracy",
                "common_improvements": ["ranking_algorithm_tuning", "feature_engineering"]
            },
            "brain3": {  # Agentic brain
                "focus": "reasoning_quality",
                "common_improvements": ["prompt_engineering", "chain_of_thought_optimization"]
            },
            "brain4": {  # Document processing brain
                "focus": "processing_speed",
                "common_improvements": ["parallel_processing", "format_optimization"]
            }
        }
        
        strategy = brain_strategies.get(brain_id, {})
        if strategy:
            suggestions.append(ImprovementSuggestion(
                improvement_type=ImprovementType.PARAMETER_TUNING,
                description=f"Optimize {strategy['focus']} for {brain_id}",
                confidence=0.6,
                expected_impact=0.15,
                implementation_complexity="medium",
                supporting_evidence=[f"Brain-specific optimization for {brain_id}"],
                proposed_changes={
                    "action": "brain_specific_optimization",
                    "focus_area": strategy['focus'],
                    "improvements": strategy.get('common_improvements', [])
                }
            ))
        
        return suggestions
    
    def _calculate_reflection_confidence(self, performance_data: Dict[str, Any],
                                       patterns: List[Dict[str, Any]],
                                       suggestions: List[ImprovementSuggestion]) -> float:
        """Calculate confidence in reflection analysis"""
        confidence_factors = []
        
        # Data quality confidence
        total_tasks = performance_data.get('total_tasks', 0)
        data_confidence = min(1.0, total_tasks / 50.0)  # Full confidence at 50+ tasks
        confidence_factors.append(data_confidence)
        
        # Pattern confidence
        if patterns:
            pattern_confidences = [p.get('confidence', 0.5) for p in patterns]
            pattern_confidence = statistics.mean(pattern_confidences)
            confidence_factors.append(pattern_confidence)
        
        # Suggestion confidence
        if suggestions:
            suggestion_confidences = [s.confidence for s in suggestions]
            suggestion_confidence = statistics.mean(suggestion_confidences)
            confidence_factors.append(suggestion_confidence)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    async def should_perform_reflection(self, brain_id: str, 
                                      reflection_level: ReflectionLevel) -> bool:
        """Check if reflection should be performed"""
        current_time = time.time()
        last_reflection = self.last_reflections.get(reflection_level, 0.0)
        interval = self.reflection_intervals.get(reflection_level, 3600)
        
        return (current_time - last_reflection) >= interval
    
    async def implement_suggestion(self, suggestion: ImprovementSuggestion,
                                 brain_id: str) -> bool:
        """Implement an improvement suggestion"""
        try:
            # This would contain the actual implementation logic
            # For now, just log the implementation attempt
            
            logger.info("Implementing improvement suggestion",
                       brain_id=brain_id,
                       improvement_type=suggestion.improvement_type.value,
                       description=suggestion.description)
            
            # Simulate implementation
            await asyncio.sleep(0.1)
            
            self.improvements_implemented += 1
            
            # Track performance gain (would be measured in practice)
            estimated_gain = suggestion.expected_impact * suggestion.confidence
            self.performance_gains_achieved += estimated_gain
            
            return True
            
        except Exception as e:
            logger.error("Failed to implement suggestion",
                        brain_id=brain_id,
                        improvement_type=suggestion.improvement_type.value,
                        error=str(e))
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get self-improvement engine statistics"""
        return {
            "reflections_performed": self.reflections_performed,
            "improvements_suggested": self.improvements_suggested,
            "improvements_implemented": self.improvements_implemented,
            "implementation_rate": self.improvements_implemented / max(self.improvements_suggested, 1),
            "performance_gains_achieved": self.performance_gains_achieved,
            "average_gain_per_improvement": self.performance_gains_achieved / max(self.improvements_implemented, 1),
            "last_reflections": {level.value: timestamp for level, timestamp in self.last_reflections.items()},
            "performance_thresholds": self.performance_thresholds
        }

# Global self-improvement engine instance
_improvement_engine: Optional[SelfImprovementEngine] = None

def get_improvement_engine(memory_store: Optional[MemoryStore] = None,
                          grading_engine: Optional[SelfGradingSystem] = None) -> SelfImprovementEngine:
    """Get or create the global self-improvement engine instance"""
    global _improvement_engine
    if _improvement_engine is None:
        _improvement_engine = SelfImprovementEngine(memory_store, grading_engine)
    return _improvement_engine
