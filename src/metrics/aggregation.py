"""Metrics aggregation logic."""

from typing import List, Dict
import statistics

from src.evaluators.base import EvaluationResult, EvaluatorType
from .types import AggregatedMetrics, CategoryScores


class MetricsAggregator:
    """Aggregates evaluation results into final metrics."""
    
    # Weights for final composite score
    CATEGORY_WEIGHTS = {
        'correctness': 0.40,
        'performance': 0.20,
        'code_quality': 0.15,
        'algorithm': 0.15,
        'petsc': 0.10,
    }
    
    # Mapping of evaluator names to categories
    EVALUATOR_CATEGORY_MAP = {
        # Correctness
        'numerical_accuracy': 'correctness',
        
        # Performance
        'execution_time': 'performance',
        
        # Code quality
        'readability': 'code_quality',
        'code_style': 'code_quality',
        'documentation': 'code_quality',
        
        # Algorithm
        'algorithm_appropriateness': 'algorithm',
        'solver_choice': 'algorithm',
        
        # PETSc
        'petsc_best_practices': 'petsc',
        'error_handling': 'petsc',
        'parallel_awareness': 'petsc',
    }
    
    # Tier thresholds
    TIER_THRESHOLDS = {
        'GOLD': 85,
        'SILVER': 70,
        'BRONZE': 50,
    }
    
    def aggregate(self, results: List[EvaluationResult]) -> AggregatedMetrics:
        """Aggregate all evaluation results.
        
        Args:
            results: List of evaluation results from all evaluators
        
        Returns:
            AggregatedMetrics with composite scores and tier
        """
        # Separate by type
        gates = [r for r in results if r.evaluator_type == EvaluatorType.GATE]
        metrics = [r for r in results if r.evaluator_type == EvaluatorType.METRIC]
        quality = [r for r in results if r.evaluator_type == EvaluatorType.QUALITY]
        
        # Check gate status
        all_gates_passed = all(
            r.passed for r in gates if r.passed is not None
        )
        gates_passed = sum(1 for r in gates if r.passed)
        gates_total = len(gates)

        # If gates failed, return early with score of 0
        if not all_gates_passed:
            return AggregatedMetrics(
                category_scores=CategoryScores(),
                composite_score=0.0,
                overall_tier='FAIL',
                all_gates_passed=False,
                gates_passed=gates_passed,
                gates_total=gates_total,
                total_evaluators=len(results),
                passed_evaluators=sum(1 for r in results if r.passed),
                failed_evaluators=sum(1 for r in results if r.passed is False),
                evaluation_results=results,
                total_evaluation_time_ms=sum(r.execution_time_ms for r in results),
                llm_evaluations_count=sum(1 for r in results if 'llm' in r.evaluation_method),
            )
        
        # Calculate category scores
        category_scores = self._calculate_category_scores(results)
        
        # Calculate composite score
        composite_score = sum(
            getattr(category_scores, cat) * weight
            for cat, weight in self.CATEGORY_WEIGHTS.items()
        )
        
        # Determine tier
        tier = self._determine_tier(composite_score, all_gates_passed)
        
        # Count evaluations
        passed_count = sum(1 for r in results if r.passed)
        failed_count = sum(1 for r in results if r.passed is False)
        llm_count = sum(1 for r in results if 'llm' in r.evaluation_method)
        total_time = sum(r.execution_time_ms for r in results)
        
        return AggregatedMetrics(
            category_scores=category_scores,
            composite_score=composite_score,
            overall_tier=tier,
            all_gates_passed=all_gates_passed,
            gates_passed=gates_passed,
            gates_total=gates_total,
            total_evaluators=len(results),
            passed_evaluators=passed_count,
            failed_evaluators=failed_count,
            evaluation_results=results,
            total_evaluation_time_ms=total_time,
            llm_evaluations_count=llm_count,
        )
    
    def _calculate_category_scores(self, results: List[EvaluationResult]) -> CategoryScores:
        """Calculate scores for each category."""
        category_values: Dict[str, List[float]] = {
            'correctness': [],
            'performance': [],
            'code_quality': [],
            'algorithm': [],
            'petsc': [],
            'semantic': [],
        }
        
        for result in results:
            # Get the appropriate score value
            if result.normalized_score is not None:
                score = result.normalized_score  # 0-1 from metrics
            elif result.quality_score is not None:
                score = result.quality_score  # 0-1 from quality
            else:
                continue  # Skip if no score available
            
            # Map to category
            category = self.EVALUATOR_CATEGORY_MAP.get(result.evaluator_name)
            if category and category in category_values:
                # Weight by confidence if available
                if result.confidence is not None and result.confidence > 0:
                    # Store tuples of (score, confidence) for weighted average
                    category_values[category].append((score, result.confidence))
                else:
                    category_values[category].append((score, 1.0))
        
        # Calculate weighted averages for each category
        scores = CategoryScores()
        
        for category, values in category_values.items():
            if values:
                # Weighted average
                total_weight = sum(conf for _, conf in values)
                if total_weight > 0:
                    weighted_sum = sum(score * conf for score, conf in values)
                    avg_score = weighted_sum / total_weight
                else:
                    avg_score = sum(score for score, _ in values) / len(values)
                
                # Convert to 0-100 scale
                setattr(scores, category, avg_score * 100)
            else:
                # No evaluators for this category
                setattr(scores, category, 0.0)
        
        return scores
    
    def _determine_tier(self, composite_score: float, gates_passed: bool) -> str:
        """Determine tier based on composite score."""
        if not gates_passed:
            return 'FAIL'
        
        if composite_score >= self.TIER_THRESHOLDS['GOLD']:
            return 'GOLD'
        elif composite_score >= self.TIER_THRESHOLDS['SILVER']:
            return 'SILVER'
        elif composite_score >= self.TIER_THRESHOLDS['BRONZE']:
            return 'BRONZE'
        else:
            return 'FAIL'
    
    def get_detailed_breakdown(self, results: List[EvaluationResult]) -> Dict[str, any]:
        """Get detailed breakdown of all evaluations."""
        breakdown = {
            'gates': [],
            'metrics': [],
            'quality': [],
        }
        
        for result in results:
            item = {
                'name': result.evaluator_name,
                'passed': result.passed,
                'score': result.quality_score or result.normalized_score or result.raw_value,
                'confidence': result.confidence,
                'feedback': result.feedback,
                'method': result.evaluation_method,
            }
            
            if result.evaluator_type == EvaluatorType.GATE:
                breakdown['gates'].append(item)
            elif result.evaluator_type == EvaluatorType.METRIC:
                breakdown['metrics'].append(item)
            elif result.evaluator_type == EvaluatorType.QUALITY:
                breakdown['quality'].append(item)
        
        return breakdown