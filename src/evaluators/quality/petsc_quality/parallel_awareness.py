"""Parallel awareness quality evaluator."""

import time
from typing import Any, Dict, Optional

from ...base import Evaluator, EvaluatorType, EvaluationResult


class ParallelAwarenessQuality(Evaluator):
    """Evaluates MPI/parallel awareness in code.
    
    Checks for proper handling of parallel execution.
    """
    
    @property
    def name(self) -> str:
        return "parallel_awareness"
    
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
        """Evaluate parallel awareness."""
        start_time = time.time()
        
        score = 0.5  # Base score
        features = []
        
        # Check for MPI-related code
        if 'PETSC_COMM_WORLD' in code:
            score += 0.2
            features.append('Uses PETSC_COMM_WORLD')
        
        # Check for rank-aware operations
        if 'MPI_Comm_rank' in code or 'PetscMPIInt' in code:
            score += 0.15
            features.append('Rank-aware code')
        
        # Check for proper parallel matrix/vector setup
        if 'MatSetSizes' in code or 'VecSetSizes' in code:
            score += 0.15
            features.append('Parallel-aware sizing')
        
        score = min(1.0, score)
        
        feedback = f"Parallel features: {', '.join(features)}" if features else "Limited parallel awareness"
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=score >= 0.7,
            quality_score=score,
            confidence=0.8,
            feedback=feedback,
            metadata={'features': features},
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )