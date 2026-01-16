import time
from typing import Any, Dict, Optional

from ..base import Evaluator, EvaluatorType, EvaluationResult


class ExecutionTimeMetric(Evaluator):
    """Measures execution time performance.
    
    This metric evaluates performance based on absolute execution time.
    Faster execution yields higher scores.
    """
    
    @property
    def name(self) -> str:
        return "execution_time"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.METRIC
    
    @property
    def evaluation_method(self) -> str:
        return "deterministic"
    
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Measure execution time performance.
        
        Args:
            code: The generated code (not used directly)
            problem: Problem specification
            execution_result: Must contain 'execution_time_sec'
        
        Returns:
            EvaluationResult with raw_value=time, normalized_score based on performance tiers
        """
        start_time = time.time()
        
        if execution_result is None or 'execution_time_sec' not in execution_result:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=False,
                raw_value=None,
                normalized_score=0.0,
                confidence=1.0,
                feedback="No execution time available",
                metadata={},
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        actual_time = execution_result['execution_time_sec']

        # Performance tiers (configurable)
        excellent_time = self.config.get('excellent_time_sec', 1.0) if self.config else 1.0
        good_time = self.config.get('good_time_sec', 5.0) if self.config else 5.0
        acceptable_time = self.config.get('acceptable_time_sec', 15.0) if self.config else 15.0
        max_time = self.config.get('max_time_sec', 60.0) if self.config else 60.0
        
        # Calculate normalized score (0.0 to 1.0)
        # Use piecewise linear scoring with performance tiers
        if actual_time <= excellent_time:
            normalized_score = 1.0
            performance_tier = "excellent"
        elif actual_time <= good_time:
            # Linear interpolation between excellent and good
            normalized_score = 0.8 + 0.2 * (good_time - actual_time) / (good_time - excellent_time)
            performance_tier = "good"
        elif actual_time <= acceptable_time:
            # Linear interpolation between good and acceptable
            normalized_score = 0.6 + 0.2 * (acceptable_time - actual_time) / (acceptable_time - good_time)
            performance_tier = "acceptable"
        elif actual_time <= max_time:
            # Linear interpolation between acceptable and max
            normalized_score = 0.2 + 0.4 * (max_time - actual_time) / (max_time - acceptable_time)
            performance_tier = "poor"
        else:
            # Beyond max time - very low score
            normalized_score = max(0.0, 0.2 * max_time / actual_time)
            performance_tier = "very poor"

        # Determine pass/fail
        passed = bool(actual_time <= max_time)

        # Generate feedback
        feedback_map = {
            "excellent": f"Excellent performance: {actual_time:.3f}s (< {excellent_time:.1f}s)",
            "good": f"Good performance: {actual_time:.3f}s ({excellent_time:.1f}s - {good_time:.1f}s)",
            "acceptable": f"Acceptable performance: {actual_time:.3f}s ({good_time:.1f}s - {acceptable_time:.1f}s)",
            "poor": f"Poor performance: {actual_time:.3f}s ({acceptable_time:.1f}s - {max_time:.1f}s)",
            "very poor": f"Very poor performance: {actual_time:.3f}s (> {max_time:.1f}s)",
        }
        feedback = feedback_map[performance_tier]
        print(feedback)
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=passed,
            raw_value=actual_time,
            normalized_score=normalized_score,
            confidence=1.0,
            feedback=feedback,
            metadata={
                'actual_time_sec': actual_time,
                'performance_tier': performance_tier,
                'excellent_time_sec': excellent_time,
                'good_time_sec': good_time,
                'acceptable_time_sec': acceptable_time,
                'max_time_sec': max_time,
                'memory_mb': execution_result.get('memory_mb'),
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )