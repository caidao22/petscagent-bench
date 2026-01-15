"""Memory safety gate - checks for memory leaks and errors."""

import time
from typing import Any, Dict, Optional

from ..base import Evaluator, EvaluatorType, EvaluationResult


class MemorySafetyGate(Evaluator):
    """Checks for memory safety issues using valgrind or similar tools.
    
    This gate checks for:
    - Memory leaks
    - Invalid memory access
    - Use of uninitialized values
    """
    
    @property
    def name(self) -> str:
        return "memory_safety"
    
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
        """Check memory safety.
        
        Args:
            code: The generated code (not used directly)
            problem: Problem specification (not used for memory check)
            execution_result: Should contain 'valgrind_output' or similar
        
        Returns:
            EvaluationResult with passed=True if no memory issues found
        """
        start_time = time.time()
        
        if execution_result is None:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=None,  # Can't determine without execution
                feedback="No execution result provided - memory safety check skipped",
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Check if valgrind was run
        valgrind_output = execution_result.get('valgrind_output')
        stderr = execution_result.get('stderr', '')
        
        if valgrind_output is None:
            # Valgrind not run - do basic checks on stderr
            has_memory_issue = any([
                'memory leak' in stderr.lower(),
                'invalid read' in stderr.lower(),
                'invalid write' in stderr.lower(),
                'segmentation fault' in stderr.lower(),
            ])
            
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=not has_memory_issue,
                confidence=0.7,  # Lower confidence without valgrind
                feedback="Basic memory safety check (valgrind not available)" if not has_memory_issue 
                        else "Potential memory safety issue detected",
                metadata={
                    'valgrind_available': False,
                    'stderr_check': True,
                },
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Parse valgrind output
        memory_safe = self._parse_valgrind_output(valgrind_output)
        
        if memory_safe:
            feedback = "No memory leaks or errors detected (valgrind)"
        else:
            feedback = "Memory safety issues detected (see metadata for details)"
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            passed=memory_safe,
            confidence=1.0,  # High confidence with valgrind
            feedback=feedback,
            metadata={
                'valgrind_available': True,
                'valgrind_output': valgrind_output[:500],  # Truncate
            },
            evaluation_method=self.evaluation_method,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def _parse_valgrind_output(self, output: str) -> bool:
        """Parse valgrind output to determine if memory is safe.
        
        Args:
            output: Valgrind output text
        
        Returns:
            True if no memory issues, False otherwise
        """
        # Look for the summary line
        # Example: "ERROR SUMMARY: 0 errors from 0 contexts"
        if 'ERROR SUMMARY: 0 errors' in output:
            return True
        
        # Look for specific issues
        memory_issues = [
            'definitely lost',
            'indirectly lost',
            'possibly lost',
            'Invalid read',
            'Invalid write',
            'Use of uninitialised',
        ]
        
        for issue in memory_issues:
            if issue in output:
                return False
        
        # If we can't determine clearly, be conservative
        return 'All heap blocks were freed' in output or 'no leaks are possible' in output