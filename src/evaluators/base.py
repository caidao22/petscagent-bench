"""Base classes for the evaluation system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EvaluatorType(Enum):
    """Type of evaluation being performed."""
    
    GATE = "gate"           # Binary checks (must pass)
    METRIC = "metric"       # Continuous measurements
    QUALITY = "quality"     # Quality assessments


@dataclass
class EvaluationResult:
    """Result from a single evaluator.
    
    This unified result supports all evaluator types:
    - Gates: Use 'passed' field
    - Metrics: Use 'raw_value' and 'normalized_score'
    - Quality: Use 'quality_score'
    """
    evaluator_name: str
    evaluator_type: EvaluatorType
    
    # For GATE evaluators
    passed: Optional[bool] = None
    
    # For METRIC evaluators
    raw_value: Optional[float] = None  # Actual measurement (e.g., 2.5 seconds, 1e-8 error)
    normalized_score: Optional[float] = None  # 0-1 normalized score
    
    # For QUALITY evaluators
    quality_score: Optional[float] = None  # 0-1 score
    
    # Common fields
    confidence: Optional[float] = None  # How confident (0-1), especially for LLM
    feedback: str = ""  # Human-readable explanation
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional data
    
    # Tracking
    evaluation_method: str = "deterministic"  # deterministic, llm_gpt4, manual, etc.
    execution_time_ms: float = 0.0


class Evaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this evaluator."""
        pass
    
    @property
    @abstractmethod
    def evaluator_type(self) -> EvaluatorType:
        """Type of evaluation this performs."""
        pass
    
    @property
    def evaluation_method(self) -> str:
        """How evaluation is performed (deterministic, llm, manual, etc.)."""
        return "deterministic"
    
    @abstractmethod
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate the generated code.
        
        Args:
            code: The generated code to evaluate
            problem: The problem specification/description
            execution_result: Results from compilation/execution (if available)
                Expected keys:
                - compiles: bool
                - compile_errors: str
                - runs: bool
                - runtime_errors: str
                - stdout: str
                - stderr: str
                - exit_code: int
                - execution_time_sec: float
                - memory_mb: float
        
        Returns:
            EvaluationResult with appropriate fields filled
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.evaluator_type.value})"