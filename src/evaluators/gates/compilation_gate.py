"""Compilation gate - checks if code compiles without errors."""

import time
from typing import Any, Dict, Optional

from ..base import Evaluator, EvaluatorType, EvaluationResult


class CompilationGate(Evaluator):
    """Checks if code compiles without errors.
    
    This is a hard gate - code MUST compile successfully to be considered valid.
    Compilation is a prerequisite for execution.
    """
    
    @property
    def name(self) -> str:
        return "compilation"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.GATE
    
    @property
    def evaluation_method(self) -> str:
        return "deterministic"
    
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Check compilation status.
        
        Args:
            code: The generated code (not used directly)
            problem: Problem specification (not used for compilation check)
            execution_result: Must contain 'compiles' key
        
        Returns:
            EvaluationResult with passed=True if compiled successfully
        """
        start_time = time.time()
        
        if execution_result is None:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=False,
                feedback="No execution result provided",
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        compiles = execution_result.get('compiles', False)
        
        if compiles:
            feedback = "Code compiled successfully"
        else:
            feedback = "Compilation failed"
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=compiles,
            confidence=1.0,  # Deterministic check
            feedback=feedback,
            metadata={
                'compiles': compiles,
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )