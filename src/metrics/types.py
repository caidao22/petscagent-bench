"""Type definitions for metrics."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.evaluators.base import EvaluationResult


@dataclass
class CategoryScores:
    """Scores broken down by category."""
    correctness: float = 0.0  # 0-100
    performance: float = 0.0  # 0-100
    code_quality: float = 0.0  # 0-100
    algorithm: float = 0.0  # 0-100
    petsc: float = 0.0  # 0-100


@dataclass
class AggregatedMetrics:
    """Aggregated metrics from all evaluators."""
    
    # Category scores (0-100)
    category_scores: CategoryScores
    
    # Overall metrics
    composite_score: float  # 0-100, weighted average
    overall_tier: str  # GOLD/SILVER/BRONZE/FAIL
    
    # Gate status
    all_gates_passed: bool
    gates_passed: int
    gates_total: int
    
    # Evaluation summary
    total_evaluators: int
    passed_evaluators: int
    failed_evaluators: int
    
    # Individual results
    evaluation_results: List[EvaluationResult] = field(default_factory=list)
    
    # Performance metrics
    total_evaluation_time_ms: float = 0.0
    llm_evaluations_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'composite_score': self.composite_score,
            'overall_tier': self.overall_tier,
            'category_scores': {
                'correctness': self.category_scores.correctness,
                'performance': self.category_scores.performance,
                'code_quality': self.category_scores.code_quality,
                'algorithm': self.category_scores.algorithm,
                'petsc': self.category_scores.petsc,
            },
            'gates': {
                'all_passed': self.all_gates_passed,
                'passed': self.gates_passed,
                'total': self.gates_total,
            },
            'summary': {
                'total_evaluators': self.total_evaluators,
                'passed': self.passed_evaluators,
                'failed': self.failed_evaluators,
            },
            'performance': {
                'total_time_ms': self.total_evaluation_time_ms,
                'llm_evaluations': self.llm_evaluations_count,
            },
        }
    
    def get_summary_string(self) -> str:
        """Get human-readable summary."""
        return f"""
{'='*60}
Evaluation Summary
{'='*60}
Overall Tier: {self.overall_tier}
Composite Score: {self.composite_score:.1f}/100

Category Scores:
  Correctness:  {self.category_scores.correctness:.1f}/100
  Performance:  {self.category_scores.performance:.1f}/100
  Code Quality: {self.category_scores.code_quality:.1f}/100
  Algorithm:    {self.category_scores.algorithm:.1f}/100
  PETSc Usage:  {self.category_scores.petsc:.1f}/100

Gates: {self.gates_passed}/{self.gates_total} passed
Evaluators: {self.passed_evaluators}/{self.total_evaluators} passed
{'='*60}
"""