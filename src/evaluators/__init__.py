"""
Evaluation system for PETSc code generation benchmark.

This module provides a comprehensive evaluation framework for assessing
generated PETSc code across multiple dimensions:
- Gates: Binary pass/fail checks (must all pass)
- Metrics: Continuous measurements (numerical accuracy, performance)
- Quality: Subjective assessments (code quality, algorithm appropriateness)
"""

from .base import (
    Evaluator,
    EvaluatorType,
    EvaluationResult,
    EvaluationConfig,
)
from .pipeline import EvaluationPipeline

__all__ = [
    'Evaluator',
    'EvaluatorType',
    'EvaluationResult',
    'EvaluationConfig',
    'EvaluationPipeline',
]