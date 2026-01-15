"""Execution time metric - measures performance."""

import time
from typing import Any, Dict, Optional

from ..base import Evaluator, EvaluatorType, EvaluationResult


class ExecutionTimeMetric(Evaluator):
    """Measures execution time and compares to baseline.
    
    This metric evaluates performance by comparing actual execution time
    to a baseline (if available).
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
            problem: May contain 'baseline_time_sec' for comparison
            execution_result: Must contain 'execution_time_sec'
        
        Returns:
            EvaluationResult with raw_value=time, normalized_score based on baseline
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
        baseline_time = problem.get('baseline_time_sec')
        
        # If no baseline, use a reasonable default or just report the time
        if baseline_time is None:
            # No baseline - just report the time
            # Score based on absolute time: faster is better
            # Use a heuristic: 1 second or less = 1.0, exponential decay after that
            target_time = self.config.get('target_time_sec', 1.0)
            normalized_score = min(1.0, target_time / max(actual_time, 0.001))
            
            feedback = f"Execution time: {actual_time:.3f}s (no baseline available)"
            passed = actual_time <= target_time * 2  # Within 2x of target
            
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=passed,
                raw_value=actual_time,
                normalized_score=normalized_score,
                confidence=0.7,  # Lower confidence without baseline
                feedback=feedback,
                metadata={
                    'actual_time_sec': actual_time,
                    'baseline_available': False,
                    'target_time_sec': target_time,
                },
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # With baseline - compute speedup
        speedup = baseline_time / max(actual_time, 0.001)  # Avoid division by zero
        
        # Normalized score:
        # - speedup >= 1.0 (faster or equal): score = 1.0
        # - speedup < 1.0 (slower): score decreases
        # - Cap at 1.0 if faster
        normalized_score = min(1.0, speedup)
        
        # Determine if passed
        max_slowdown = self.config.get('max_slowdown_factor', 2.0)
        passed = actual_time <= baseline_time * max_slowdown
        
        # Generate feedback
        if speedup >= 1.0:
            feedback = f"Good performance: {actual_time:.3f}s (baseline: {baseline_time:.3f}s, speedup: {speedup:.2f}x)"
        elif speedup >= 0.5:
            feedback = f"Acceptable performance: {actual_time:.3f}s (baseline: {baseline_time:.3f}s, slowdown: {1/speedup:.2f}x)"
        else:
            feedback = f"Poor performance: {actual_time:.3f}s (baseline: {baseline_time:.3f}s, slowdown: {1/speedup:.2f}x)"
        
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
                'baseline_time_sec': baseline_time,
                'speedup': speedup,
                'slowdown': 1 / speedup if speedup > 0 else float('inf'),
                'baseline_available': True,
                'memory_mb': execution_result.get('memory_mb'),
                'solver_iterations': execution_result.get('solver_iterations'),
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )