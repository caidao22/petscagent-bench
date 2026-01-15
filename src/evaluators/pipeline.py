"""Evaluation pipeline orchestrator."""

import asyncio
from typing import List, Dict, Any, Optional

from .base import Evaluator, EvaluatorType, EvaluationResult

# Import all evaluators
from .gates import (
    CompilationGate,
    ExecutionGate,
    MemorySafetyGate,
    APIUsageGate,
)
from .metrics import (
    NumericalAccuracyMetric,
    ExecutionTimeMetric,
)
from .quality.code_quality import (
    ReadabilityQuality,
    CodeStyleQuality,
    DocumentationQuality,
)
from .quality.algorithm_quality import (
    AlgorithmAppropriatenessQuality,
    SolverChoiceQuality,
)
from .quality.petsc_quality import (
    PETScBestPracticesQuality,
    ErrorHandlingQuality,
    ParallelAwarenessQuality,
)


class EvaluationPipeline:
    """Orchestrates multiple evaluators in a structured pipeline.
    
    Pipeline phases:
    1. Gates (must all pass) - run in parallel
    2. Metrics (measurements) - run in parallel
    3. Quality (assessments) - run with LLM rate limiting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluation pipeline.
        
        Args:
            config: Configuration dictionary loaded from YAML/JSON
        """
        self.config = config or {}
        self.gates: List[Evaluator] = []
        self.metrics: List[Evaluator] = []
        self.quality: List[Evaluator] = []
        
        self._setup_evaluators()
    
    def _get_eval_config(self, key: str, default: Any = None) -> Any:
        """Get value from evaluation config section."""
        return self.config.get('evaluation', {}).get(key, default)
    
    def _get_llm_config(self, key: str, default: Any = None) -> Any:
        """Get value from LLM config section."""
        return self.config.get('evaluation', {}).get('llm', {}).get(key, default)
    
    def _setup_evaluators(self):
        """Initialize evaluators based on configuration."""
        
        # Gates (always enabled - these are critical)
        if self._get_eval_config('enable_gates', True):
            self.gates = [
                CompilationGate(),
                ExecutionGate(),
                MemorySafetyGate(),
                APIUsageGate(),
            ]
        
        # Metrics (deterministic measurements)
        if self._get_eval_config('enable_metrics', True):
            self.metrics = [
                NumericalAccuracyMetric(),
                ExecutionTimeMetric(),
            ]
        
        # Quality evaluators
        if self._get_eval_config('enable_quality', True):
            llm_config = {
                'llm_model': self._get_llm_config('model', 'gpt-4o-mini'),
                'llm_temperature': self._get_llm_config('temperature', 0.3),
                'use_llm': True,
            }
            
            self.quality = [
                # Code quality (can use LLM or static analysis)
                ReadabilityQuality(llm_config),
                CodeStyleQuality(llm_config),
                DocumentationQuality(llm_config),
                
                # Algorithm quality (LLM-based)
                AlgorithmAppropriatenessQuality(llm_config),
                SolverChoiceQuality(llm_config),
                
                # PETSc quality (mixed)
                PETScBestPracticesQuality(llm_config),
                ErrorHandlingQuality(),  # Deterministic
                ParallelAwarenessQuality(),  # Deterministic
            ]
    
    async def evaluate(
        self,
        code: str,
        problem: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """Run all evaluators and return results.
        
        Args:
            code: The generated code to evaluate
            problem: Problem specification
            execution_result: Results from compilation/execution
        
        Returns:
            List of evaluation results from all evaluators
        """
        all_results: List[EvaluationResult] = []
        parallel = self._get_eval_config('parallel_evaluation', True)
        
        # Phase 1: Gates (must all pass to continue)
        print("Phase 1: Running gate evaluators...")
        if parallel:
            gate_results = await asyncio.gather(*[
                gate.evaluate(code, problem, execution_result)
                for gate in self.gates
            ])
        else:
            gate_results = []
            for gate in self.gates:
                result = await gate.evaluate(code, problem, execution_result)
                gate_results.append(result)
        
        all_results.extend(gate_results)
        
        # Check if all gates passed
        gates_passed = all(
            r.passed for r in gate_results 
            if r.passed is not None
        )
        
        if not gates_passed:
            print("Gate(s) failed. Skipping remaining evaluation.")
            return all_results
        
        # Phase 2: Metrics (parallel - all deterministic)
        print("Phase 2: Running metric evaluators...")
        if parallel:
            metric_results = await asyncio.gather(*[
                metric.evaluate(code, problem, execution_result)
                for metric in self.metrics
            ])
        else:
            metric_results = []
            for metric in self.metrics:
                result = await metric.evaluate(code, problem, execution_result)
                metric_results.append(result)
        all_results.extend(metric_results)
        
        # Phase 3: Quality (rate-limited for LLM-based)
        print("Phase 3: Running quality evaluators...")
        
        # Separate LLM-based from deterministic
        llm_quality = [q for q in self.quality if 'llm' in q.evaluation_method]
        non_llm_quality = [q for q in self.quality if 'llm' not in q.evaluation_method]
        # Non-LLM quality (can run in parallel, fast)
        if non_llm_quality:
            if parallel:
                non_llm_results = await asyncio.gather(*[
                    q.evaluate(code, problem, execution_result)
                    for q in non_llm_quality
                ])
            else:
                non_llm_results = []
                for q in non_llm_quality:
                    result = await q.evaluate(code, problem, execution_result)
                    non_llm_results.append(result)
            
            all_results.extend(non_llm_results)
        
        # LLM quality (rate-limited)
        if llm_quality:
            max_concurrent = self._get_llm_config('max_concurrent_calls', 3)
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def rate_limited_eval(evaluator: Evaluator) -> EvaluationResult:
                async with semaphore:
                    return await evaluator.evaluate(code, problem, execution_result)
            
            llm_results = await asyncio.gather(*[
                rate_limited_eval(q) for q in llm_quality
            ], return_exceptions=True)
            
            # Filter out exceptions
            for result in llm_results:
                if isinstance(result, Exception):
                    print(f"LLM evaluation error: {result}")
                else:
                    all_results.append(result)
        
        print(f"Evaluation complete: {len(all_results)} evaluators ran")
        return all_results
    
    def add_evaluator(self, evaluator: Evaluator):
        """Add a custom evaluator to the pipeline.
        
        Args:
            evaluator: Custom evaluator instance
        """
        if evaluator.evaluator_type == EvaluatorType.GATE:
            self.gates.append(evaluator)
        elif evaluator.evaluator_type == EvaluatorType.METRIC:
            self.metrics.append(evaluator)
        elif evaluator.evaluator_type == EvaluatorType.QUALITY:
            self.quality.append(evaluator)
    
    def get_evaluator_count(self) -> Dict[str, int]:
        """Get count of evaluators by type."""
        return {
            'gates': len(self.gates),
            'metrics': len(self.metrics),
            'quality': len(self.quality),
            'total': len(self.gates) + len(self.metrics) + len(self.quality),
        }