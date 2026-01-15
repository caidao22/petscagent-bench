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
            problem: Must contain 'test_cases' with 'expected_output'
            execution_result: Must contain 'stdout' with solution data
        
        Returns:
            EvaluationResult with raw_value=error_norm, normalized_score based on error
        """
        start_time = time.time()
        
        # Check if test cases with expected output are available
        if 'test_cases' not in problem or not problem['test_cases']:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=None,
                raw_value=None,
                normalized_score=None,
                confidence=None,
                feedback="No test cases available for comparison",
                metadata={'reference_available': False},
                evaluation_method=self.evaluation_method,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get the expected output from the first test case
        test_case = problem['test_cases'][0]
        if 'expected_output' not in test_case:
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type=self.evaluator_type,
                passed=None,
                raw_value=None,
                normalized_score=None,
                confidence=None,
                feedback="No expected output in test case",
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
        expected_output = test_case['expected_output']
        
        try:
            error_norm = self._compute_error_norm(stdout, expected_output)
            
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
    
    def _compute_error_norm(self, stdout: str, expected_output: Any) -> float:
        """Compute error norm between output and expected result.
        
        Args:
            stdout: Program output containing solution (last N lines should contain N numbers)
            expected_output: Expected output (array of N numbers)
        
        Returns:
            Error norm (float)
        """
        # Convert expected_output to numpy array
        if isinstance(expected_output, (list, tuple)):
            reference_values = np.array(expected_output, dtype=float)
        elif isinstance(expected_output, np.ndarray):
            reference_values = expected_output.astype(float)
        elif isinstance(expected_output, (int, float)):
            reference_values = np.array([float(expected_output)])
        else:
            raise ValueError(f"Unsupported expected_output type: {type(expected_output)}")
        
        # Extract the last N lines from stdout, where N is the length of expected_output
        lines = stdout.strip().split('\n')
        n_expected = len(reference_values)
        
        # Get the last N lines and try to extract numbers from them
        last_n_lines = lines[-n_expected:] if len(lines) >= n_expected else lines
        
        solution_values = []
        for line in last_n_lines:
            # Extract numbers from each line
            numbers = self._extract_numbers(line)
            if len(numbers) > 0:
                # Take the first number from each line (or could take last, depending on format)
                solution_values.append(numbers[0])
        
        solution_values = np.array(solution_values, dtype=float)
        
        if len(solution_values) == 0:
            raise ValueError("No numerical values found in output")
        
        # Ensure we have the right number of values
        if len(solution_values) != len(reference_values):
            raise ValueError(
                f"Number of output values ({len(solution_values)}) does not match "
                f"expected output length ({len(reference_values)})"
            )
        
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