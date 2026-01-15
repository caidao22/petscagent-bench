"""Error handling quality evaluator."""

import time
from typing import Any, Dict, Optional

from ...base import Evaluator, EvaluatorType, EvaluationResult


class ErrorHandlingQuality(Evaluator):
    """Evaluates proper use of PETSc error handling (CHKERRQ).
    
    This is a deterministic check - counts CHKERRQ usage.
    """
    
    @property
    def name(self) -> str:
        return "error_handling"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.QUALITY
    
    @property
    def evaluation_method(self) -> str:
        return "deterministic"
    
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate error handling."""
        start_time = time.time()
        
        # Count CHKERRQ usage
        chkerrq_count = code.count('CHKERRQ')
        
        # Count PETSc function calls (rough estimate)
        petsc_calls = sum([
            code.count('Petsc'),
            code.count('Vec'),
            code.count('Mat'),
            code.count('KSP'),
            code.count('SNES'),
        ])
        
        # Ideal: roughly one CHKERRQ per PETSc call
        if petsc_calls > 0:
            ratio = chkerrq_count / petsc_calls
            # Score based on ratio
            if ratio >= 0.8:
                score = 1.0
                feedback = f"Excellent error handling: {chkerrq_count} CHKERRQ calls"
            elif ratio >= 0.5:
                score = 0.8
                feedback = f"Good error handling: {chkerrq_count} CHKERRQ calls"
            elif ratio >= 0.3:
                score = 0.6
                feedback = f"Acceptable error handling: {chkerrq_count} CHKERRQ calls"
            else:
                score = 0.3
                feedback = f"Poor error handling: only {chkerrq_count} CHKERRQ calls"
        else:
            score = 0.5
            feedback = "No PETSc calls detected"
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=score >= 0.7,
            quality_score=score,
            confidence=1.0,  # Deterministic
            feedback=feedback,
            metadata={
                'chkerrq_count': chkerrq_count,
                'petsc_calls_estimate': petsc_calls,
                'ratio': chkerrq_count / petsc_calls if petsc_calls > 0 else 0,
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )