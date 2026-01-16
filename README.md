# PETSc Agent Benchmark

An agentified evaluation framework for testing PETSc code generation agents using A2A (Agent-to-Agent) and MCP (Model Context Protocol) standards.

## Overview

This project implements a multi-agent system for evaluating code generation agents that produce PETSc (Portable, Extensible Toolkit for Scientific Computation) programs. The system uses:

- **A2A Protocol**: Standardized agent-to-agent communication
- **MCP Protocol**: Tool access for compilation and execution
- **Multi-tier Evaluation**: Gates, metrics, and quality assessments
- **Automated Benchmarking**: End-to-end evaluation workflow

### Why PETSc?

PETSc represents a **uniquely challenging domain** for code generation agents, making it an ideal benchmark for evaluating LLM capabilities in scientific computing:

**🔬 Scientific Computing Complexity**
- **Domain Expertise Required**: Demands deep understanding of numerical methods, PDEs, linear algebra, and computational science
- **Mathematical Rigor**: Correct equation discretization, stability analysis, and convergence requirements
- **Performance Critical**: Solutions must be numerically accurate AND computationally efficient

**🏗️ Software Engineering Challenges**
- **Large API Surface**: 1000+ functions across diverse components (TS, SNES, KSP, TAO, DM, Vec, Mat)
- **Complex Abstractions**: Multi-level solver hierarchies
- **Error Handling Discipline**: Mandatory error checking (PetscCall) throughout
- **Configuration Complexity**: Runtime options database, solver selection, performance tuning

**⚡ Parallel Computing Demands**
- **MPI Programming**: Distributed data structures, domain decomposition, parallel communication
- **Scalability Concerns**: Load balancing, ghost values, efficient parallel I/O
- **Hardware Awareness**: GPU acceleration (CUDA/HIP backends)

**📚 Documentation and Best Practices**
- **Style Requirements**: Specific PETSc naming conventions and code organization
- **Modern Standards**: Evolution from PetscCall error handling
- **Performance Patterns**: Optimal usage of viewers, options, logging, and monitoring

**🎯 Real-World Impact**

PETSc is used in production by thousands of researchers and engineers across:
- Climate modeling and weather prediction
- Computational fluid dynamics
- Material science and chemistry
- Astrophysics and cosmology
- Subsurface flow and reservoir simulation
- Structural mechanics and optimization
See https://petsc.org/main/miscellaneous/applications_publications/ for more details

**Why This Matters for LLM Evaluation**:

1. **Beyond Toy Problems**: PETSc benchmarks test whether LLMs can handle real scientific software, not just algorithmic puzzles

2. **Correctness is Verifiable**: Unlike creative tasks, scientific computations have ground truth - wrong answers are measurable

3. **Multi-Dimensional Quality**: Success requires simultaneously achieving:
   - Mathematical correctness
   - Software engineering best practices
   - Performance efficiency
   - Parallel scalability

4. **Transferable Skills**: Mastery of PETSc code generation indicates capability for other complex scientific/engineering domains (deal.II, Trilinos, hypre, SUNDIALS)

5. **High Stakes**: Errors in scientific software can lead to incorrect research conclusions, failed simulations, or wasted supercomputer time

By benchmarking on PETSc, we evaluate whether LLMs can truly assist in **mission-critical scientific computing** - not just generate syntactically correct code, but produce scientifically valid, performant, and maintainable solutions.

## Architecture

The system consists of three main components:

1. **Green Agent** (Assessment Manager)
   - Loads benchmark problems from the dataset
   - Distributes tasks to the Purple Agent
   - Evaluates generated code through a comprehensive pipeline
   - Generates detailed assessment reports

2. **Purple Agent** (Target Under Test)
   - Receives problem descriptions via A2A
   - Generates PETSc C/C++ code using an LLM
   - Returns code and CLI arguments
   - Isolated from evaluation logic

3. **MCP Server** (Tool Provider)
   - Provides compilation tools (make)
   - Provides execution tools (run with arguments)
   - Manages PETSc environment configuration

## Project Structure

```
├── data/                           # Benchmark problem datasets
│   └── problems_test.jsonl         # Test problem specifications
├── main.py                         # CLI entry point
├── pyproject.toml                  # Python project configuration
├── README.md                       # This file
├── src/
│   ├── green_agent/                # Assessment manager agent
│   │   ├── agent.py               # Core evaluation logic
│   │   ├── server.py              # A2A server implementation
│   │   └── mcp_client.py          # MCP client for tools
│   ├── purple_agent/               # Target agent being tested
│   │   └── petsc_agent.py         # Code generation agent
│   ├── evaluators/                 # Evaluation system
│   │   ├── base.py                # Base evaluator classes
│   │   ├── pipeline.py            # Evaluation orchestrator
│   │   ├── gates/                 # Binary pass/fail checks
│   │   ├── metrics/               # Quantitative measurements
│   │   └── quality/               # Quality assessments
│   ├── metrics/                    # Metrics aggregation
│   │   ├── aggregation.py         # Score computation
│   │   └── types.py               # Metric data types
│   ├── util/                       # Utility modules
│   │   ├── a2a_comm.py            # A2A communication helpers
│   │   └── llm_client.py          # LLM client utilities
│   └── launcher.py                # Evaluation coordinator
├── config/                         # Configuration files
│   └── evaluation_config.yaml     # Evaluation pipeline config
├── output/                         # Generated reports and results
├── generated_codes/                # Code generated by Purple Agent
└── purple_agent_cache/             # Cached responses (optional)
```

## Installation

### Prerequisites

1. **PETSc Installation**: Install PETSc from [https://petsc.org/](https://petsc.org/) for local testing only
2. **Python 3.11+**: Required for the evaluation framework
3. **UV Package Manager**: Install from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

### Setup

1. Install dependencies using `uv`:

```bash
uv sync
```

2. Create a `.env` file in the root directory with the following variables:

```bash
# LLM API Keys (at least one required)
GEMINI_API_KEY="<your_gemini_key>"
OPENAI_API_KEY="<your_openai_key>"

# PETSc Configuration (required for compilation/execution)
PETSC_DIR="<path_to_petsc_installation>"
PETSC_ARCH="<petsc_architecture>"  # e.g., arch-darwin-c-debug
```

## Usage

### Quick Start

Launch the complete evaluation workflow:

```bash
uv run main.py launch
```

This command will:
1. Start the Green Agent (assessment manager)
2. Start the Purple Agent (code generator)
3. Start the MCP server (compilation/execution tools)
4. Run all benchmark problems
5. Generate evaluation reports in `output/`

### Individual Components

You can also run components separately for development/debugging:

```bash
# Start only the Green Agent
uv run main.py green

# Start only the Purple Agent
uv run main.py purple
```

### Configuration

The evaluation pipeline can be configured via `config/evaluation_config.yaml`:

```yaml
evaluation:
  enable_gates: true          # Enable binary pass/fail checks
  enable_metrics: true        # Enable quantitative measurements
  enable_quality: true        # Enable quality assessments
  parallel_evaluation: true   # Run evaluators in parallel
  
  llm:
    model: "openai/gpt-4o-mini"  # LLM for quality evaluation
    temperature: 0.3             # LLM temperature
    max_concurrent_calls: 3      # Rate limiting for LLM calls
  
  thresholds:
    min_llm_confidence: 0.7      # Minimum confidence for LLM evaluations

scoring:
  weights:
    correctness: 0.35     # Weight for correctness score
    performance: 0.15     # Weight for performance metrics
    code_quality: 0.15    # Weight for code quality
    algorithm: 0.15       # Weight for algorithm choice
    petsc: 0.10          # Weight for PETSc best practices
    semantic: 0.10       # Weight for semantic correctness
  
  tiers:
    gold: 85      # Minimum score for GOLD tier
    silver: 70    # Minimum score for SILVER tier
    bronze: 50    # Minimum score for BRONZE tier
```

## Evaluation System

The evaluation pipeline consists of three phases:

### Phase 1: Gates (Must Pass)

Binary checks that code must pass to be considered valid:

- **Compilation Gate**: Code must compile without errors
- **Execution Gate**: Code must run without crashes
- **Memory Safety Gate**: No memory leaks or invalid accesses
- **API Usage Gate**: Correct PETSc API usage

If any gate fails, evaluation stops and the code receives a FAIL tier.

### Phase 2: Metrics (Measurements)

Quantitative measurements of code performance:

- **Numerical Accuracy**: Correctness of numerical results
- **Execution Time**: Runtime performance

### Phase 3: Quality (Assessments)

Qualitative assessments of code quality:

**Code Quality:**
- Readability: Code structure and naming
- Code Style: Adherence to C/C++ conventions
- Documentation: Comments and explanations

**Algorithm Quality:**
- Algorithm Appropriateness: Suitability for the problem
- Solver Choice: Optimal PETSc solver selection

**PETSc Quality:**
- Best Practices: PETSc usage patterns
- Error Handling: Proper error checking
- Parallel Awareness: MPI and parallel considerations

## Benchmark Problems

The benchmark suite is designed to comprehensively test PETSc code generation across **diverse computational domains**, **varying difficulty levels**, and **different PETSc components**. All problems support **MPI parallelization** and the framework is **easily extensible** with custom problems.

### Key Features

✨ **Diversity**: Covers ODEs, PDEs, and optimization across multiple scientific domains  
🎯 **Difficulty Range**: From medium (advection) to high (stiff systems) complexity  
🔧 **Extensibility**: Simple JSON format for adding new benchmark problems  
⚡ **MPI-Ready**: All problems can be executed in parallel with multiple MPI ranks  
🧩 **Component Coverage**: Tests TS (time-stepping), TAO (optimization), Vec, Mat, and more

### Current Benchmark Suite

### 1. Robertson ODE (Stiff System)

**Problem Type**: Time-dependent Ordinary Differential Equations  
**PETSc Components**: TS (Time Stepper)  
**Difficulty**: High (stiff system with multiple time scales)

**Description**:  
A classic benchmark for stiff ODE solvers modeling a chemical reaction system:

```
dy₁/dt = -0.04y₁ + 10⁴y₂y₃
dy₂/dt = 0.04y₁ - 10⁴y₂y₃ - 3×10⁷y₂²
dy₃/dt = 3×10⁷y₂²
```

**Initial Conditions**: y₁=1, y₂=0, y₃=0 at t=0  
**Time Range**: [0, 100]  
**Expected Solution**: y₁≈0.617, y₂≈5.61×10⁻⁶, y₃≈0.383

**Key Challenges**:
- Extreme stiffness (time scales differ by ~7 orders of magnitude)
- Requires implicit time stepping (Crank-Nicolson)
- Careful tolerance selection for accuracy
- Tests IFunction/IJacobian implementation

**Test Configuration**:
```bash
-ts_type cn -ts_time_step 1e-7 -ts_adapt_type basic -ts_exact_final_time matchstep
```

---

### 2. 1D Advection PDE

**Problem Type**: Time-dependent Partial Differential Equation  
**PETSc Components**: TS (Time Stepper), Vec, DA (Distributed Array)  
**Difficulty**: Medium (hyperbolic PDE with periodic boundaries)

**Description**:  
Linear advection equation modeling wave propagation:

```
∂u/∂t + c∂u/∂x = 0
```

**Domain**: x ∈ [0,1] with periodic boundary conditions  
**Initial Condition**: u(x,0) = sin(2πx)  
**Time Range**: [0, 1]  
**Advection Speed**: c (constant)

**Key Challenges**:
- Spatial discretization (first-order upwind scheme)
- Periodic boundary condition handling
- Uniform grid management
- Stability constraints (CFL condition)
- Tests RHSFunction for explicit time stepping

**Test Configuration**:
```bash
-ts_type rk -ts_rk_type 4
```

---

### 3. Rosenbrock Optimization (Banana Function)

**Problem Type**: Unconstrained Nonlinear Optimization  
**PETSc Components**: TAO (Toolkit for Advanced Optimization)  
**Difficulty**: Medium (narrow curved valley, ill-conditioned)

**Description**:  
Classic optimization benchmark with a narrow, banana-shaped valley:

```
f(x,y) = (1-x)² + 100(y-x²)²
```

**Global Minimum**: (x,y) = (1,1) with f(1,1) = 0

**Key Challenges**:
- Highly ill-conditioned (valley is 100× narrower than wide)
- Tests gradient computation accuracy
- Requires quasi-Newton methods (LMVM)
- Convergence monitoring
- Modern PetscCall() error handling style

**Test Configuration**:
```bash
-tao_view -tao_monitor
```

---

### Problem Dataset Format

Benchmark problems are stored in JSON format in the `data/` directory:

```json
{
  "problem_name": "Problem_Name",
  "problem_id": "unique_id",
  "problem_description": "Detailed problem specification...",
  "test_cases": [
    {
      "args": "-solver_options",
      "expected_output": [numerical_values]
    }
  ]
}
```

### Evaluation Criteria

Each problem is evaluated across multiple dimensions:

1. **Correctness** (35%): Numerical accuracy, equation implementation, boundary conditions
2. **Algorithm Choice** (15%): Solver selection, discretization method appropriateness
3. **Code Quality** (15%): Readability, conventions, documentation
4. **Performance** (15%): Execution time, convergence rate
5. **PETSc Best Practices** (10%): Runtime configurability, error handling
6. **Semantic Correctness** (10%): Physics preservation, stability

## Output

Evaluation results are saved to the `output/` directory:

- `benchmark_summary.json`: Overall statistics and per-problem results
- `evaluation_report.txt`: Human-readable summary report
- `evaluation_detailed_report.json`: Detailed scores and feedback
- `benchmark_result_<problem_name>.json`: Individual problem results

### Tier System

Codes are assigned to tiers based on composite scores:

- 🥇 **GOLD** (≥85): Excellent code quality and correctness
- 🥈 **SILVER** (≥70): Good code with minor issues
- 🥉 **BRONZE** (≥50): Functional but needs improvement
- ❌ **FAIL** (<50 or gate failure): Significant issues

## Development

### Adding Custom Evaluators

To add a new evaluator:

1. Create a new evaluator class inheriting from `Evaluator`
2. Implement required properties: `name`, `evaluator_type`, `evaluation_method`
3. Implement the `evaluate()` method
4. Add to the pipeline in `src/evaluators/pipeline.py`

Example:

```python
from src.evaluators.base import Evaluator, EvaluatorType, EvaluationResult

class MyCustomEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "my_custom_check"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.QUALITY
    
    async def evaluate(self, code, problem, execution_result):
        # Your evaluation logic here
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            quality_score=0.8,
            feedback="Custom evaluation passed"
        )
```

### Caching

The Green Agent supports caching Purple Agent responses to speed up development:

```python
# In src/green_agent/agent.py
agent = Agent(
    purple_agent_url=purple_url,
    mcp_server_url=mcp_url,
    use_cache=True  # Enable caching
)
```

Cached responses are stored in `purple_agent_cache/`.

## Troubleshooting

### Common Issues

1. **PETSc not found**: Ensure `PETSC_DIR` and `PETSC_ARCH` are set correctly in `.env`
2. **LLM API errors**: Verify API keys are valid and have sufficient quota
3. **Agent timeout**: Increase timeout in `src/util/a2a_comm.py` if needed
4. **Port conflicts**: Modify ports in `src/launcher.py` if defaults are in use
