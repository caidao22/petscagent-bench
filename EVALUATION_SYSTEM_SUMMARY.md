# PETSc Code Evaluation System - Implementation Summary (Updated)

## Overview

A comprehensive evaluation framework for assessing generated PETSc code with **14 evaluators** across 3 evaluation types, **fully integrated** into the Green Agent benchmarking pipeline.

## Architecture

### Three-Tier Evaluation Model

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: GATES (Must Pass)                                │
│  ├── Compilation          (deterministic)                  │
│  ├── Execution            (deterministic)                  │
│  ├── Memory Safety        (deterministic)                  │
│  └── API Usage            (deterministic)                  │
│                                                             │
│  Phase 2: METRICS (Measurements)                           │
│  ├── Numerical Accuracy   (deterministic)                  │
│  └── Execution Time       (deterministic)                  │
│                                                             │
│  Phase 3: QUALITY (Assessments)                            │
│  ├── Code Quality         (LLM or static)                  │
│  │   ├── Readability                                       │
│  │   ├── Code Style                                        │
│  │   └── Documentation                                     │
│  ├── Algorithm Quality    (LLM)                            │
│  │   ├── Algorithm Appropriateness                         │
│  │   └── Solver Choice                                     │
│  └── PETSc Quality        (mixed)                          │
│      ├── Best Practices   (LLM)                            │
│      ├── Error Handling   (deterministic)                  │
│      └── Parallel Aware   (deterministic)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   METRICS AGGREGATOR                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Category Scores (0-100):                                  │
│  ├── Correctness  (35% weight)                            │
│  ├── Performance  (15% weight)                            │
│  ├── Code Quality (15% weight)                            │
│  ├── Algorithm    (15% weight)                            │
│  ├── PETSc Usage  (10% weight)                            │
│  └── Semantic     (10% weight)                            │
│                                                             │
│  Composite Score = Σ(category × weight)                   │
│                                                             │
│  Tier Assignment:                                          │
│  ├── GOLD   (≥85)                                         │
│  ├── SILVER (≥70)                                         │
│  ├── BRONZE (≥50)                                         │
│  └── FAIL   (<50 or gates failed)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Complete Metrics List (14 Total)

### 1. Gates (4) - Binary Pass/Fail

| Metric | What | Method | Critical |
|--------|------|--------|----------|
| **Compilation** | Code compiles successfully | Deterministic | ✅ |
| **Execution** | Runs without crash | Deterministic | ✅ |
| **Memory Safety** | No leaks/errors (Valgrind) | Deterministic | ✅ |
| **API Usage** | PetscInit/Finalize present | Static Analysis | ✅ |

**If ANY gate fails → Overall score = 0 (FAIL)**

### 2. Metrics (2) - Continuous Measurements

| Metric | Raw Value | Normalized Score | Method |
|--------|-----------|------------------|--------|
| **Numerical Accuracy** | Error norm | exp(-error/tol) | Deterministic |
| **Execution Time** | Seconds | baseline/actual | Deterministic |

### 3. Quality (8) - Subjective Assessments

#### Code Quality (3)
| Metric | Assessment | Default Method | Configurable |
|--------|------------|----------------|---------------|
| **Readability** | Variable names, structure | LLM | ✅ Static option |
| **Code Style** | PETSc/C conventions | LLM | ✅ Static option |
| **Documentation** | Comments, clarity | LLM | ✅ Static option |

#### Algorithm Quality (2)
| Metric | Assessment | Method |
|--------|------------|--------|
| **Algorithm Appropriateness** | Suitable approach | LLM |
| **Solver Choice** | KSP/SNES type | LLM |

#### PETSc Quality (3)
| Metric | Assessment | Method |
|--------|------------|--------|
| **Best Practices** | CLI options, viewers | LLM |
| **Error Handling** | CHKERRQ usage | Deterministic |
| **Parallel Awareness** | MPI-aware code | Deterministic |

## Implementation Structure

```
petscagent_bench/
├── src/
│   ├── evaluators/
│   │   ├── __init__.py                # Exports
│   │   ├── base.py                    # Base classes, enums
│   │   ├── pipeline.py                # Orchestration
│   │   ├── README.md                  # Documentation
│   │   ├── gates/                     # 4 gate evaluators
│   │   │   ├── __init__.py
│   │   │   ├── compilation_gate.py
│   │   │   ├── execution_gate.py
│   │   │   ├── memory_safety_gate.py
│   │   │   └── api_usage_gate.py
│   │   ├── metrics/                   # 2 metric evaluators
│   │   │   ├── __init__.py
│   │   │   ├── numerical_accuracy.py
│   │   │   └── execution_time.py
│   │   └── quality/                   # 8 quality evaluators
│   │       ├── __init__.py
│   │       ├── code_quality/          # 3 evaluators
│   │       │   ├── __init__.py
│   │       │   ├── readability.py
│   │       │   ├── code_style.py
│   │       │   └── documentation.py
│   │       ├── algorithm_quality/     # 2 evaluators
│   │       │   ├── __init__.py
│   │       │   ├── algorithm_appropriateness.py
│   │       │   └── solver_choice.py
│   │       └── petsc_quality/         # 3 evaluators
│   │           ├── __init__.py
│   │           ├── best_practices.py
│   │           ├── error_handling.py
│   │           └── parallel_awareness.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── types.py                   # Data structures
│   │   └── aggregation.py             # Scoring logic
│   ├── util/
│   │   └── llm_client.py              # OpenAI wrapper
│   └── green_agent/
│       └── agent.py                   # ✅ INTEGRATED
├── config/
│   └── evaluation_config.yaml         # Configuration
└── examples/
    └── evaluation_example.py          # Usage example
```

## Key Design Decisions

### 1. Why 3 Types (Gates/Metrics/Quality)?

**Semantic types instead of method-based classification:**

- **Gates**: Must-pass requirements (compilation, execution, memory safety, API usage)
- **Metrics**: Objective measurements (time, error)
- **Quality**: Subjective assessments (readability, appropriateness)

This separates **WHAT** we evaluate from **HOW** we evaluate it.

### 2. Why Both Deterministic and LLM?

| Use Deterministic When | Use LLM When |
|------------------------|---------------|
| ✅ Objective facts (compiles? runs?) | ✅ Subjective quality (readable?) |
| ✅ Measurable (time, error) | ✅ Semantic understanding (correct approach?) |
| ✅ Ground truth available | ✅ No reference solution |
| ✅ Fast & free | ✅ Complex reasoning needed |

**Rule**: If you can measure it objectively, don't use LLM.

### 3. Scoring Formula

```python
if not all_gates_passed:
    score = 0
else:
    score = (
        0.35 × correctness +      # Numerical accuracy
        0.15 × performance +       # Execution time
        0.15 × code_quality +      # Readability, style, docs
        0.15 × algorithm +         # Algorithm, solver
        0.10 × petsc +             # Best practices
        0.10 × semantic            # (Reserved for future use)
    )
```

## Full Green Agent Integration

The evaluation system is **fully integrated** into the Green Agent's benchmarking pipeline:

### Enhanced BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    problem_name: str
    problem_id: str
    runs: bool
    time_used_sec: float
    compiles: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    cli_args: Optional[str] = None
    
    # Evaluation fields (NEW)
    composite_score: Optional[float] = None  # 0-100
    tier: Optional[str] = None  # GOLD/SILVER/BRONZE/FAIL
    category_scores: Optional[Dict[str, float]] = None
    evaluation_summary: Optional[Dict[str, Any]] = None
    evaluation_details: Optional[List[Dict[str, Any]]] = None
```

### Agent Initialization

```python
class Agent():
    def __init__(self, purple_agent_url, mcp_server_url, max_num_prob=None, use_cache=True):
        # ... existing setup ...
        
        # Initialize evaluation system with config from file or defaults
        eval_config = load_evaluation_config()
        self.evaluation_pipeline = EvaluationPipeline(eval_config)
        self.metrics_aggregator = MetricsAggregator(eval_config)
        
        print(f"✅ Evaluation system initialized with {self.evaluation_pipeline.get_evaluator_count()['total']} evaluators")
```

### Evaluation Workflow

```python
async def run(self, message, updater):
    for idx, data in enumerate(test_data[:limit]):
        # 1. Get code from purple agent (with caching)
        # 2. Compile and run code
        # 3. NEW: Evaluate code
        
        if generated_codes:
            await self._evaluate_code(benchmark_result, data, generated_codes)
        
        # 4. Update summary with tier distribution
        if br.tier:
            summary["tier_distribution"][br.tier] += 1
    
    # 5. Generate comprehensive evaluation report
    await self._create_evaluation_report(results, summary, updater)
```

### Private Evaluation Method

```python
async def _evaluate_code(
    self,
    benchmark_result: BenchmarkResult,
    problem_data: Dict[str, Any],
    generated_codes: List[str],
) -> None:
    """Run evaluation pipeline on generated codes."""
    
    # Prepare execution result
    execution_result = {
        'compiles': benchmark_result.compiles,
        'runs': benchmark_result.runs,
        'stdout': benchmark_result.stdout or '',
        'stderr': benchmark_result.stderr or '',
        'execution_time_sec': benchmark_result.time_used_sec,
        'memory_mb': None,
    }
    
    # Run evaluation pipeline
    eval_results = await self.evaluation_pipeline.evaluate(
        code=generated_codes[0],
        problem=problem_data,
        execution_result=execution_result
    )
    
    # Aggregate results
    aggregated = self.metrics_aggregator.aggregate(eval_results)
    
    # Update benchmark result
    benchmark_result.composite_score = aggregated.composite_score
    benchmark_result.tier = aggregated.overall_tier
    benchmark_result.category_scores = {
        'correctness': aggregated.category_scores.correctness,
        'performance': aggregated.category_scores.performance,
        'code_quality': aggregated.category_scores.code_quality,
        'algorithm': aggregated.category_scores.algorithm,
        'petsc': aggregated.category_scores.petsc,
    }
```

## Configuration System

### YAML Configuration (config/evaluation_config.yaml)

```yaml
evaluation:
  # Enable/disable evaluation phases
  enable_gates: true
  enable_metrics: true
  enable_quality: true
  
  # LLM settings for quality evaluators
  llm:
    model: "openai/gpt-4o-mini"
    temperature: 0.3
    max_concurrent_calls: 3  # Rate limiting
  
  # Performance settings
  parallel_evaluation: true
  
  # LLM Thresholds
  thresholds:
    min_llm_confidence: 0.7

# Scoring configuration
scoring:
  # Category weights (must sum to 1.0)
  weights:
    correctness: 0.35
    performance: 0.15
    code_quality: 0.15
    algorithm: 0.15
    petsc: 0.10
    semantic: 0.10
  
  # Tier thresholds (0-100 scale)
  tiers:
    gold: 85
    silver: 70
    bronze: 50
```

### Configuration Loading

Supports multiple formats with graceful fallback:

```python
def load_evaluation_config(config_path: str = "config/evaluation_config.yaml") -> Dict[str, Any]:
    """Load evaluation configuration from file or use defaults.
    
    Supports both JSON and YAML formats. Format is auto-detected by file extension.
    Falls back to sensible defaults if config file not found.
    """
    config_file = Path(config_path)
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            print(f"✅ Loaded evaluation config from {config_path}")
            return config_data
        except Exception as e:
            print(f"⚠️ Failed to load config: {e}")
    
    # Fall back to defaults
    return { /* default config */ }
```

## Output Format

### Console Output During Evaluation

```
@@@ Green agent: ✅ Evaluation system initialized with 14 evaluators
[1/3] Running Advection_PDE...
@@@ Green agent: ✅ Loaded cached response for Advection_PDE
@@@ Green agent: Compile and run the code...
@@@ Green agent: Evaluating generated code...
Phase 1: Running gate evaluators...
Phase 2: Running metric evaluators...
Phase 3: Running quality evaluators...
Evaluation complete: 14 evaluators ran
@@@ Green agent: ✅ Evaluation complete: Score=87.5, Tier=GOLD
```

### Text Report (evaluation_report.txt)

```
================================================================================
EVALUATION REPORT
================================================================================

Total Problems: 3
Successful Executions: 3
Failed Executions: 0
Average Execution Time: 2.45s

Average Composite Score: 76.3/100

Tier Distribution:
  🥇 GOLD:   1 (33.3%)
  🥈 SILVER: 1 (33.3%)
  🥉 BRONZE: 1 (33.3%)
  ❌ FAIL:   0 (0.0%)

================================================================================
PER-PROBLEM RESULTS
================================================================================

🥇 Advection_PDE (Score: 87.5/100)
   Correctness: 92.0, Performance: 85.0, Code Quality: 78.0

🥈 Robertson_ODE (Score: 73.2/100)
   Correctness: 80.0, Performance: 70.0, Code Quality: 68.0

🥉 Rosenbrock_banana_function (Score: 68.1/100)
   Correctness: 75.0, Performance: 65.0, Code Quality: 60.0
```

### JSON Output (output/benchmark_summary.json)

```json
{
  "summary": {
    "total": 3,
    "runs_count": 3,
    "failure_count": 0,
    "avg_time_sec": 2.45,
    "avg_composite_score": 76.3,
    "tier_distribution": {
      "GOLD": 1,
      "SILVER": 1,
      "BRONZE": 1,
      "FAIL": 0
    }
  },
  "results": [
    {
      "problem_name": "Advection_PDE",
      "problem_id": "adv_001",
      "runs": true,
      "time_used_sec": 2.1,
      "compiles": true,
      "composite_score": 87.5,
      "tier": "GOLD",
      "category_scores": {
        "correctness": 92.0,
        "performance": 85.0,
        "code_quality": 78.0,
        "algorithm": 88.0,
        "petsc": 82.0
      },
      "evaluation_summary": {
        "total_evaluators": 14,
        "passed_evaluators": 13,
        "failed_evaluators": 1,
        "all_gates_passed": true,
        "gates_passed": 4,
        "gates_total": 4
      },
      "evaluation_details": [
        {
          "name": "compilation",
          "type": "gate",
          "method": "deterministic",
          "passed": true,
          "score": null,
          "raw_value": null,
          "confidence": 1.0,
          "feedback": "Code compiled successfully"
        },
        {
          "name": "numerical_accuracy",
          "type": "metric",
          "method": "deterministic",
          "passed": null,
          "score": 0.95,
          "raw_value": 1.2e-8,
          "confidence": 1.0,
          "feedback": "Excellent numerical accuracy"
        }
      ]
    }
  ]
}
```

### Detailed Evaluation Report (evaluation_detailed_report.json)

```json
{
  "summary": {
    "total": 3,
    "avg_composite_score": 76.3,
    "tier_distribution": { "GOLD": 1, "SILVER": 1, "BRONZE": 1, "FAIL": 0 }
  },
  "per_problem_scores": [
    {
      "problem_name": "Advection_PDE",
      "problem_id": "adv_001",
      "tier": "GOLD",
      "composite_score": 87.5,
      "category_scores": {
        "correctness": 92.0,
        "performance": 85.0,
        "code_quality": 78.0,
        "algorithm": 88.0,
        "petsc": 82.0
      },
      "evaluation_summary": {
        "total_evaluators": 14,
        "passed_evaluators": 13,
        "failed_evaluators": 1,
        "all_gates_passed": true
      }
    }
  ]
}
```

## Purple Agent Caching System

The Green Agent includes a **caching system** to avoid redundant Purple Agent calls:

```python
def __init__(self, purple_agent_url, mcp_server_url, max_num_prob=None, use_cache=True):
    self.use_cache = use_cache
    self.cache_dir = Path("./purple_agent_cache")
    self.cache_dir.mkdir(exist_ok=True)

def _get_cache_path(self, problem_name: str) -> Path:
    """Get the cache file path for a given problem."""
    safe_name = re.sub(r'[^\w\-_]', '_', problem_name)
    return self.cache_dir / f"{safe_name}.pkl"

def _load_cached_response(self, problem_name: str):
    """Load cached purple agent response if it exists."""
    cache_path = self._get_cache_path(problem_name)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def _save_cached_response(self, problem_name: str, response):
    """Save purple agent response to cache."""
    cache_path = self._get_cache_path(problem_name)
    with open(cache_path, 'wb') as f:
        pickle.dump(response, f)
```

**Benefits:**
- 🚀 Faster re-evaluation during development
- 💰 No redundant Purple Agent calls (cost savings)
- 🔄 Consistent results for testing evaluation changes
- 📁 Stored in `./purple_agent_cache/` as `.pkl` files

## Performance & Cost

### Evaluation Time

| Configuration | Time | Cost (per problem) |
|---------------|------|--------------------| 
| Gates + Metrics only | ~200ms | $0.00 |
| + Quality (static) | ~500ms | $0.00 |
| + Quality (LLM mini) | ~30-45s | ~$0.01-0.02 |
| + Quality (LLM gpt-4o) | ~30-45s | ~$0.10-0.15 |

### Optimization Strategies

1. **Development/Testing**: 
   ```yaml
   enable_quality: false  # Skip LLM evaluations
   ```

2. **Production**: 
   ```yaml
   llm:
     model: "openai/gpt-4o-mini"  # Good quality/cost ratio
   ```

3. **Research/High Quality**: 
   ```yaml
   llm:
     model: "openai/gpt-4o"  # Best quality
   ```

4. **Hybrid Approach**: 
   ```yaml
   readability:
     use_llm: false  # Use static analysis
   code_style:
     use_llm: false
   # Keep LLM for algorithm quality
   ```

## Usage Examples

### Running the Full Benchmark

```bash
# Run with evaluation enabled (default)
python main.py

# Or use the launcher
python src/launcher.py
```

### Quick Test Without LLM

Edit `config/evaluation_config.yaml`:
```yaml
evaluation:
  enable_quality: false  # Fast mode
```

### Standalone Evaluation

```python
from src.evaluators import EvaluationPipeline
from src.metrics import MetricsAggregator
from src.green_agent.agent import load_evaluation_config

# Initialize
config = load_evaluation_config()
pipeline = EvaluationPipeline(config)
aggregator = MetricsAggregator(config)

# Prepare execution result
execution_result = {
    'compiles': True,
    'runs': True,
    'stdout': '...',
    'execution_time_sec': 2.5,
}

# Evaluate
results = await pipeline.evaluate(code, problem_data, execution_result)
metrics = aggregator.aggregate(results)

print(f"Score: {metrics.composite_score:.1f}/100")
print(f"Tier: {metrics.overall_tier}")
```

### Custom Evaluator

```python
from src.evaluators.base import Evaluator, EvaluatorType, EvaluationResult

class CustomQualityEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "custom_metric"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.QUALITY
    
    async def evaluate(self, code, problem, execution_result):
        # Your evaluation logic here
        score = 0.85  # 0-1 scale
        
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            quality_score=score,
            confidence=0.9,
            feedback="Custom evaluation passed",
            evaluation_method="custom"
        )

# Add to pipeline
pipeline.add_evaluator(CustomQualityEvaluator())
```

## Files in Repository (Complete Implementation)

### Evaluators (14 files)

```
src/evaluators/
├── __init__.py                          # ✅ Exports
├── base.py                              # ✅ Base classes, EvaluatorType enum
├── pipeline.py                          # ✅ Orchestration
├── README.md                            # ✅ Documentation
├── gates/                               # ✅ 4 gate evaluators
│   ├── __init__.py
│   ├── compilation_gate.py
│   ├── execution_gate.py
│   ├── memory_safety_gate.py
│   └── api_usage_gate.py
├── metrics/                             # ✅ 2 metric evaluators
│   ├── __init__.py
│   ├── numerical_accuracy.py
│   └── execution_time.py
└── quality/                             # ✅ 8 quality evaluators
    ├── __init__.py
    ├── code_quality/
    │   ├── __init__.py
    │   ├── readability.py
    │   ├── code_style.py
    │   └── documentation.py
    ├── algorithm_quality/
    │   ├── __init__.py
    │   ├── algorithm_appropriateness.py
    │   └── solver_choice.py
    └── petsc_quality/
        ├── __init__.py
        ├── best_practices.py
        ├── error_handling.py
        └── parallel_awareness.py
```

### Supporting Infrastructure

```
src/metrics/
├── __init__.py                          # ✅ Exports
├── types.py                             # ✅ AggregatedMetrics, CategoryScores
└── aggregation.py                       # ✅ MetricsAggregator

src/util/
└── llm_client.py                        # ✅ OpenAI wrapper

src/green_agent/
└── agent.py                             # ✅ FULLY INTEGRATED

config/
└── evaluation_config.yaml               # ✅ Configuration

examples/
└── evaluation_example.py                # ✅ Usage example
```

### Generated Output

```
output/
├── benchmark_summary.json               # Main results with evaluations
├── benchmark_result_{problem}.json      # Per-problem detailed results
├── evaluation_report.txt                # Human-readable summary
└── evaluation_detailed_report.json      # Detailed evaluation breakdown

purple_agent_cache/
├── Advection_PDE.pkl                    # Cached responses
├── Robertson_ODE.pkl
└── Rosenbrock_banana_function.pkl

generated_codes/
├── Advection_PDE.c                      # Generated code files
├── Robertson_ODE.c
└── Rosenbrock_banana_function.c
```

## Production Status

✅ **Complete implementation** with all 14 evaluators

✅ **Fully integrated** with Green Agent benchmarking

✅ **Configurable** via YAML/JSON

✅ **Well-documented** with README and examples

✅ **Tested** on real PETSc problems

✅ **Caching** for efficient development

✅ **Comprehensive reporting** with multiple output formats

✅ **Production-ready** and actively running
