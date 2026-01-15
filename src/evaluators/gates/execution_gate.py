"""Execution gate - checks if code runs without crashes."""

import time
from typing import Any, Dict, Optional

from ..base import Evaluator, EvaluatorType, EvaluationResult


class ExecutionGate(Evaluator):
    """Checks if code executes without runtime errors.
    
    This is a hard gate - code MUST run successfully to be considered valid.
    """
    
    @property
    def name(self) -> str:
        return "execution"
    
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
        """Check execution status.
        
        Args:
            code: The generated code (not used directly)
            problem: Problem specification (not used for execution check)
            execution_result: Must contain 'runs', 'exit_code', 'runtime_errors' keys
        
        Returns:
            EvaluationResult with passed=True if executed successfully
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
        
        runs = execution_result.get('runs', False)
        exit_code = execution_result.get('exit_code', -1)
        runtime_errors = execution_result.get('runtime_errors', '')
        stderr = execution_result.get('stderr', '')
        
        # Check for common runtime issues
        has_segfault = 'segmentation fault' in stderr.lower() or 'sigsegv' in stderr.lower()
        has_assertion = 'assertion' in stderr.lower()
        has_abort = 'abort' in stderr.lower()
        
        if runs and exit_code == 0:
            feedback = "Code executed successfully"
        else:
            error_indicators = []
            if has_segfault:
                error_indicators.append("segmentation fault")
            if has_assertion:
                error_indicators.append("assertion failure")
            if has_abort:
                error_indicators.append("aborted")
            if exit_code != 0:
                error_indicators.append(f"exit code {exit_code}")
            
            if error_indicators:
                feedback = f"Execution failed: {', '.join(error_indicators)}"
            else:
                error_preview = runtime_errors[:200] + "..." if len(runtime_errors) > 200 else runtime_errors
                feedback = f"Execution failed: {error_preview or 'Unknown error'}"
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=runs and exit_code == 0,
            confidence=1.0,  # Deterministic check
            feedback=feedback,
            metadata={
                'exit_code': exit_code,
                'runtime_errors': runtime_errors,
                'has_segfault': has_segfault,
                'has_assertion': has_assertion,
                'has_abort': has_abort,
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )