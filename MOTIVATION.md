# The PETSc Agent Benchmark: Teaching AI to Speak the Language of Supercomputers

petscagent-bench is a benchmark and harness for evaluating AI agents that generate PETSc-based scientific codes running under real MPI/PETSc execution. It is intended for PETSc/HPC practitioners who want to understand what AI can currently do for them, and for AI researchers who need a rigorous, domain-specific benchmark for project-level code generation.

---

## The Challenge

Imagine you're a scientist trying to simulate how air circulates in the atmosphere, or an engineer modeling how heat flows through a nuclear reactor. These problems require solving massive systems of equations—millions or billions of unknowns—running on the world's most powerful supercomputers.

For 30+ years, **PETSc** (Portable, Extensible Toolkit for Scientific Computation) has been the gold standard library for this. It powers simulations at national labs, universities, and research centers worldwide. But there's a catch: **PETSc is notoriously difficult to learn**.

Writing correct, efficient PETSc code requires understanding:

- Distributed memory parallelism (MPI)
- Sparse matrix formats and solvers
- Preconditioners and convergence criteria
- Domain decomposition and load balancing
- GPU acceleration with CUDA/Kokkos

A PhD student might spend *months* just learning to write their first working simulation.

---

## The Promise of AI

What if an AI agent could write PETSc code for you? Just describe your physics problem in plain English:

> *"Solve the 2D heat equation on a unit square with Dirichlet boundary conditions, using a backward Euler time integrator and algebraic multigrid preconditioner."*

and out comes production-ready, parallel C code that runs on a supercomputer cluster.

Large language models can already generate nontrivial PETSc code from such descriptions—but that raises a critical question:

> *How do we know if the generated code is actually correct?*

---

## The Problem with Evaluating Code Agents

Traditional benchmarks for code generation (like HumanEval) test simple functions: *"Write a function to reverse a string."* You run it, check the output, done.

PETSc code is different:

- **It must compile** with complex dependencies (MPI, BLAS, PETSc itself)
- **It must run** on parallel systems without deadlocks or race conditions
- **It must be numerically correct**—a simulation that runs but gives wrong answers is worse than useless
- **It must be efficient**—a correct but slow solver defeats the purpose
- **It must follow best practices**—error handling, memory management, solver configuration

No existing benchmark captures this combination of numerical, parallel, and software-engineering requirements for PETSc-based codes.

---

## Enter the Green–Purple Architecture

We built **petscagent-bench** as a *battle arena* where AI agents prove their worth:


```
┌─────────────────┐         ┌─────────────────┐
│   GREEN AGENT   │◄───────►│  PURPLE AGENT   │
│   (The Judge)   │  A2A    │ (The Contender) │
└────────┬────────┘         └─────────────────┘
         │
         ▼
┌─────────────────┐
│   MCP SERVER    │
│ (The Execution  │
│    Chamber)     │
└─────────────────┘
```


- The **Purple Agent** is the code generator under test—e.g., GPT-4, Claude, Gemini, or your own fine-tuned model. It receives problem descriptions and produces PETSc code.
- The **Green Agent** is the impartial judge. It:
  1. Feeds problems to the Purple Agent
  2. Compiles the generated code in a real PETSc environment
  3. Runs it with MPI on actual hardware
  4. Evaluates correctness, performance, and code quality
  5. Assigns scores and tiers (🥇 Gold, 🥈 Silver, 🥉 Bronze, ❌ Fail)
- The **MCP Server** provides the execution environment—a sandboxed PETSc installation where code is compiled and run safely.

This separation of roles allows evaluation of remote/closed models without needing access to model internals.

---

## Our Contribution

### A Multi-Faceted Evaluation Framework

petscagent-bench introduces a three-stage evaluation pipeline:


```
Problem Description
        │
        ▼
┌───────────────────┐
│   Code Generation │ ◄── Purple Agent (LLM under test)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Gate Evaluation │ ◄── Compilation, Execution, Output
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Metric Evaluation │ ◄── Accuracy, Performance, Memory
└───────────────────┘
        │
        ▼
┌───────────────────┐
│Quality Evaluation │ ◄── Style, Algorithm, Best Practices
└───────────────────┘
        │
        ▼
   Composite Score + Tier
```


At a high level, the benchmark provides:

- A suite of PETSc tasks (e.g., PDEs, linear systems) with reference behavior.
- An automated pipeline that compiles and runs generated code with MPI in a PETSc environment.
- Multi-criteria scoring over numerical accuracy, performance, and code quality.
- Tiered outcomes (GOLD/SILVER/BRONZE/FAIL) for quick comparison.

### Key Innovations

**1. Agent-Based Architecture**

Unlike monolithic test harnesses, petscagent-bench uses independent agents communicating via standardized protocols (A2A, MCP). This enables:

- Evaluation of remote/proprietary models
- Fair comparison without access to model internals
- Distributed deployment across machines

**2. Real Execution Environment**

Generated code is compiled and executed in a real PETSc environment with MPI, not simulated or statically analyzed. This catches:

- Compilation errors from incorrect API usage
- Runtime failures from parallel programming bugs
- Numerical errors from incorrect implementations

**3. PETSc- and Science-Aware Scoring**

The scoring integrates multiple aspects specific to scientific PETSc codes, including

- Correctness (numerical accuracy)
- Performance (execution efficiency)
- Code Quality (readability, documentation)
- Algorithm choice (solver/method selection)
- PETSc Practices (error handling, conventions)
- Semantic fidelity (physics preservation, boundary conditions)

These criteria are combined into a composite score, which is then mapped to intuitive tiers (GOLD/SILVER/BRONZE/FAIL) for easy comparison across agents and tasks.

---

## What You Can Study with This Benchmark

petscagent-bench enables investigation of:

1. **Model Comparison**: How do different LLMs compare on scientific code generation?
2. **Capability Analysis**: Which aspects of PETSc code generation are most challenging for LLMs?
3. **Prompt Engineering**: How do different prompting strategies affect code quality?
4. **Fine-tuning Impact**: Can domain-specific fine-tuning improve scientific code generation?
5. **Failure Mode Analysis**: What types of errors do LLMs make in scientific code?

For PETSc users, this helps quantify what current AI tools can (and cannot) safely automate. For AI researchers, it provides a concrete, domain-heavy testbed beyond generic coding problems.

---

## The Bigger Picture

This benchmark is a stepping stone toward a future where:

- **Scientists describe problems in natural language** and get working simulation code
- **AI assistants help debug and optimize** existing PETSc applications
- **New researchers onboard faster** with AI-generated starter code
- **Best practices propagate automatically** through AI training

But we can only get there if we can *measure progress*. petscagent-bench aims to provide a rigorous, reproducible way to evaluate how well AI understands the complex world of high-performance scientific computing.

---

## Join the Effort

We welcome contributions in the following areas:

- **More test problems** spanning different physics domains
- **Better evaluation metrics** for numerical correctness
- **Diverse agent implementations** to benchmark
- **Community feedback** on evaluation priorities

Good benchmarks help us measure real progress and that benefits everyone working on AI for scientific computing.

---

*“In science, if you can't measure it, you can't improve it.”* — Lord Kelvin