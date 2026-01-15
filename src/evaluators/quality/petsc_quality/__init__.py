"""PETSc-specific quality evaluators."""

from .best_practices import PETScBestPracticesQuality
from .error_handling import ErrorHandlingQuality
from .parallel_awareness import ParallelAwarenessQuality

__all__ = [
    'PETScBestPracticesQuality',
    'ErrorHandlingQuality',
    'ParallelAwarenessQuality',
]
