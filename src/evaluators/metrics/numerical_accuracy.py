"""Numerical accuracy metric - measures error against reference solution."""

import time
import re
import numpy as np
from typing import Any, Dict, Optional

from ..base import Evaluator, EvaluatorType, EvaluationResult


class NumericalAccuracyMetric(Evaluator):
    """Measures numerical accuracy against reference solution.
    
    This metric computes the error norm between the computed solution
    and a reference solution (if available).
    """
    
    @property
    def name(self) -> str:
        return "numerical_accuracy"
    
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
        """Compute numerical accuracy.
        
        Args:
            code: The generated code (not used directly)
            problem: Must contain 'reference_solution' or 'expected_output'
            execution_result: Must contain 'stdout' with solution data
        
        Returns:
            EvaluationResult with raw_value=error_norm, normalized_score based on error
        """
        start_time = time.time()
        
        # Check if reference solution is available
        if 'reference_solution' not in problem and 'expected_output' not in problem:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=None,
                raw_value=None,
                normalized_score=None,
                confidence=None,
                feedback="No reference solution available for comparison",
                metadata={'reference_available': False},
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        if execution_result is None or 'stdout' not in execution_result:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=False,
                raw_value=None,
                normalized_score=0.0,
                confidence=1.0,
                feedback="No execution output available",
                metadata={'reference_available': True, 'output_available': False},
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Extract solution from output
        stdout = execution_result['stdout']
        reference = problem.get('reference_solution') or problem.get('expected_output')
        
        try:
            error_norm = self._compute_error_norm(stdout, reference)
            
            # Normalize score: use exponential decay
            # score = exp(-k * error) where k is chosen so that error=1e-6 gives score ~0.9
            # k = -ln(0.9) / 1e-6 ≈ 105361
            # For simplicity, use: score = exp(-error / tolerance)
            tolerance = self.config.get('error_tolerance', 1e-6)
            normalized_score = min(1.0, np.exp(-error_norm / tolerance))
            
            # Determine if passed based on threshold
            threshold = self.config.get('error_threshold', 1e-6)
            passed = error_norm < threshold
            
            if passed:
                feedback = f"Excellent accuracy: error = {error_norm:.2e} (threshold: {threshold:.2e})"
            elif error_norm < threshold * 100:
                feedback = f"Acceptable accuracy: error = {error_norm:.2e} (threshold: {threshold:.2e})"
            else:
                feedback = f"Poor accuracy: error = {error_norm:.2e} (threshold: {threshold:.2e})"
            
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=passed,
                raw_value=error_norm,
                normalized_score=normalized_score,
                confidence=1.0,
                feedback=feedback,
                metadata={
                    'error_norm': error_norm,
                    'threshold': threshold,
                    'tolerance': tolerance,
                    'reference_available': True,
                    'output_available': True,
                },
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=False,
                raw_value=None,
                normalized_score=0.0,
                confidence=0.5,
                feedback=f"Error computing accuracy: {str(e)}",
                metadata={
                    'error': str(e),
                    'error_type': type(e).__name__,
                },
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _compute_error_norm(self, stdout: str, reference: Any) -> float:
        """Compute error norm between output and reference.
        
        Args:
            stdout: Program output containing solution
            reference: Reference solution (can be various formats)
        
        Returns:
            Error norm (float)
        """
        # Try to extract numerical values from stdout
        solution_values = self._extract_numbers(stdout)
        
        if isinstance(reference, (list, tuple, np.ndarray)):
            reference_values = np.array(reference)
        elif isinstance(reference, (int, float)):
            reference_values = np.array([reference])
        elif isinstance(reference, str):
            reference_values = self._extract_numbers(reference)
        else:
            raise ValueError(f"Unsupported reference type: {type(reference)}")
        
        if len(solution_values) == 0:
            raise ValueError("No numerical values found in output")
        
        if len(reference_values) == 0:
            raise ValueError("No numerical values found in reference")
        
        # Ensure same length (take minimum)
        min_len = min(len(solution_values), len(reference_values))
        solution_values = solution_values[:min_len]
        reference_values = reference_values[:min_len]
        
        # Compute relative error norm
        error = np.linalg.norm(solution_values - reference_values)
        ref_norm = np.linalg.norm(reference_values)
        
        if ref_norm > 1e-14:  # Avoid division by zero
            relative_error = error / ref_norm
        else:
            relative_error = error
        
        return relative_error
    
    def _extract_numbers(self, text: str) -> np.ndarray:
        """Extract floating point numbers from text.
        
        Args:
            text: Text containing numbers
        
        Returns:
            Array of extracted numbers
        """
        # Pattern to match floating point numbers (including scientific notation)
        pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        matches = re.findall(pattern, text)
        
        if not matches:
            return np.array([])
        
        return np.array([float(m) for m in matches])