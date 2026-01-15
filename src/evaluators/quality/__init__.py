"""Quality evaluators - subjective assessments of code quality."""

# Code quality
from .code_quality.readability import ReadabilityQuality
from .code_quality.code_style import CodeStyleQuality
from .code_quality.documentation import DocumentationQuality

# Algorithm quality
from .algorithm_quality.algorithm_appropriateness import AlgorithmAppropriatenessQuality
from .algorithm_quality.solver_choice import SolverChoiceQuality

# PETSc quality
from .petsc_quality.best_practices import PETScBestPracticesQuality
from .petsc_quality.error_handling import ErrorHandlingQuality
from .petsc_quality.parallel_awareness import ParallelAwarenessQuality

__all__ = [
    # Code quality
    'ReadabilityQuality',
    'CodeStyleQuality',
    'DocumentationQuality',
    # Algorithm quality
    'AlgorithmAppropriatenessQuality',
    'SolverChoiceQuality',
    # PETSc quality
    'PETScBestPracticesQuality',
    'ErrorHandlingQuality',
    'ParallelAwarenessQuality',
]