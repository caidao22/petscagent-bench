import os
import json
import time
import re
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Message,
    TaskState,
    Part,
    TextPart,
    SendMessageSuccessResponse,
    Message,
)
from a2a.utils import get_message_text, new_agent_text_message, get_text_parts, get_file_parts
from src.util.a2a_comm import send_message
from pathlib import Path

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import dotenv

dotenv.load_dotenv()

from petsc_compile_run_mcp_client import PetscCompileRunMCPClient

# Import evaluation system
from src.evaluators import EvaluationPipeline, EvaluationConfig
from src.metrics import MetricsAggregator


def read_from_json(path):
    """Reads all the test problems from a given directory"""
    import pathlib

    if not os.path.isdir(path):
        raise RuntimeError(f"Directory {path} does not exist")
    data = []
    for file in pathlib.Path(path).iterdir():
        if not os.path.isfile(file):
            continue
        with open(file, "r", encoding="utf-8") as fd:
            data.append(json.loads(fd.read().strip()))
    return data


@dataclass
class BenchmarkResult:
    problem_name: str
    problem_id: str
    success: bool
    time_used_sec: float
    is_error: bool
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    purple_agent_text: Optional[str] = None
    file_path: Optional[str] = None
    cli_args: Optional[str] = None

    # Evaluation fields
    composite_score: Optional[float] = None  # 0-100
    tier: Optional[str] = None  # GOLD/SILVER/BRONZE/FAIL
    category_scores: Optional[Dict[str, float]] = None
    evaluation_summary: Optional[Dict[str, Any]] = None
    evaluation_details: Optional[List[Dict[str, Any]]] = None


class Agent():
    """
    This class represents a green agent that manages assessment and evaluation of test tasks.

    The agent distributes test tasks to participant agents, collects their responses, and reports the results.
    """
    def __init__(self, purple_agent_url, mcp_server_url, max_num_prob=None):
        self.purple_agent_url = purple_agent_url
        self.mcp_client = PetscCompileRunMCPClient(mcp_server_url)
        self.max_num_prob = max_num_prob
        self.metrics = {}

        # Initialize evaluation system
        eval_config = EvaluationConfig(
            enable_gates=True,
            enable_metrics=True,
            enable_quality=True,  # Set to False to disable LLM-based evaluation
            llm_model=os.getenv("EVALUATION_LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=0.3,
            max_concurrent_llm_calls=3,
            parallel_evaluation=True,
        )
        self.evaluation_pipeline = EvaluationPipeline(eval_config)
        self.metrics_aggregator = MetricsAggregator()
        print(f"✅ Evaluation system initialized with {self.evaluation_pipeline.get_evaluator_count()['total']} evaluators")

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Green agent implementation - manages assessment and evaluation.

        This Green agent distributes test tasks to participant agents and collects their response. No environment interaction or multiple steps for now.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use send_message(message, url) to call participant agents.
        """
        results: List[BenchmarkResult] = []
        summary: Dict[str, Any] = {
            "total": 0,
            "success_count": 0,
            "failure_count": 0,
            "avg_time_sec": None,
            "avg_composite_score": None,
            "tier_distribution": {"GOLD": 0, "SILVER": 0, "BRONZE": 0, "FAIL": 0},
        }

        input_text = get_message_text(message)
        data_file_path = Path("./data")
        test_data = read_from_json(data_file_path)
        limit = self.max_num_prob or len(test_data)

        for idx, data in enumerate(test_data[:limit], start=1):
            timestamp_started = time.time()
            pname = data["problem_name"]
            pid = data["problem_id"]
            pdesc = data["problem_description"]

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[{idx}/{len(test_data)}] Running {pname}..."),
            )

            br = BenchmarkResult(
                problem_name=pname,
                problem_id=pid,
                success=False,
                time_used_sec=0.0,
                is_error=False,
            )

            generated_code = None
            execution_stdout = None
            execution_stderr = None

            try:
                print(
                    f"@@@ Green agent: Sending message to purple agent... -->\n{pdesc}"
                )
                # (1) send task description to purple agent and get generated code
                purple_agent_response = await send_message(
                    self.purple_agent_url,
                    pdesc,
                    context_id=pname,
                )
                res_root = purple_agent_response.root
                assert isinstance(res_root, SendMessageSuccessResponse)
                res_result = res_root.result
                print("res_result", res_result)
                assert isinstance(
                    res_result, Message
                )  # though, a robust design should also support Task
                text_list = get_text_parts(res_result.parts)
                file_list = get_file_parts(res_result.parts)
                assert (
                    len(text_list) == 1
                ), "Expecting exactly one text part from the purple agent"
                purple_text = text_list[0]
                br.purple_agent_text = purple_text

                # Ensure output folder exists
                out_path = Path("./generated_codes")
                out_path.mkdir(exist_ok=True)

                # Parse response to find code
                _PATTERN = re.compile(
                    r"^Code generation successful[^\n]*\n"
                    r"cli_args:\s*(?P<cli_args>[^\n]+)\n",
                    re.DOTALL,
                )
                m = _PATTERN.search(purple_text)
                if not m:
                    raise ValueError(
                        "Could not parse purple agent response. Probably failed to generate the code."
                    )
                cli_args = m.group("cli_args")
                br.cli_args = cli_args
                # (2) build and run
                print(
                    f"@@@ Green agent: Compile and run the code generated by purple agent..."
                )
                await self.mcp_client.initialize()
                for f in file_list:
                    print("f type", type(f.bytes))
                    print("file name ", f.name)
                    # Save to local file
                    local_file = out_path / f.name
                    local_file.write_text(f.bytes)
                    await self.mcp_client.upload_file(filename=local_file)
                await self.mcp_client.make(executable=pname)
                (result, response) = await self.mcp_client.run_executable(
                    executable=pname, args=cli_args
                )
                execution_stdout = result
                execution_stderr = getattr(response, 'stderr', '')
                print(result)
                br.is_error = response.isError
                br.success = not br.is_error
                br.stdout = result
            except Exception as e:
                br.is_error = True
                br.success = False
                br.error_message = f"{type(e).__name__}: {e}"
                execution_stderr = str(e)
            finally:
                await self.mcp_client.finalize()
                br.time_used_sec = time.time() - timestamp_started

                # Run evaluation system
                if generated_code is not None:
                    print(f"@@@ Green agent: Evaluating generated code...")
                    await self._evaluate_code(br, data, generated_code, execution_stdout, execution_stderr)
                results.append(br)
                # Update rolling summary
                summary["total"] += 1
                if br.success:
                    summary["success_count"] += 1
                else:
                    summary["failure_count"] += 1

                # Update evaluation summary
                if br.tier:
                    summary["tier_distribution"][br.tier] += 1

                # Optional: per-case artifact (useful for debugging)
                await updater.add_artifact(
                    name=f"benchmark_result_{pname}.json",
                    parts=[TextPart(text=json.dumps(asdict(br), indent=2))],
                )

        # Final summary artifact
        times = [r.time_used_sec for r in results]
        summary["avg_time_sec"] = (sum(times) / len(times)) if times else None

        # Calculate average evaluation score
        scores = [r.composite_score for r in results if r.composite_score is not None]
        summary["avg_composite_score"] = (sum(scores) / len(scores)) if scores else None

        # Save output to file
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        local_path = output_dir / "benchmark_summary.json"
        json_data = {
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        local_path.write_text(json.dumps(json_data, indent=2))
        await updater.add_artifact(
            name="benchmark_summary.json",
            parts=[TextPart(text=json.dumps(json_data, indent=2))],
            metadata=summary,
        )

        # # Create evaluation summary report
        await self._create_evaluation_report(results, summary, updater)

        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(

                f"Done. {summary['success_count']}/{summary['total']} succeeded. "
                f"Avg score: {summary.get('avg_composite_score', 0):.1f}/100"
            ),
        )

    async def _evaluate_code(
        self,
        benchmark_result: BenchmarkResult,
        problem_data: Dict[str, Any],
        generated_code: str,
        stdout: Optional[str],
        stderr: Optional[str]
    ) -> None:
        """Run evaluation pipeline on generated code.

        Args:
            benchmark_result: BenchmarkResult to update with evaluation metrics
            problem_data: Original problem specification
            generated_code: The generated C code
            stdout: Program output
            stderr: Error output
        """
        try:
            # Prepare execution result for evaluators
            execution_result = {
                'compiles': not benchmark_result.is_error,
                'compile_errors': benchmark_result.error_message if benchmark_result.is_error else '',
                'runs': benchmark_result.success,
                'runtime_errors': stderr or '',
                'exit_code': 0 if benchmark_result.success else 1,
                'stdout': stdout or '',
                'stderr': stderr or '',
                'execution_time_sec': benchmark_result.time_used_sec,
                'memory_mb': None,  # TODO: Add memory tracking if available
            }

            # Run evaluation pipeline
            eval_results = await self.evaluation_pipeline.evaluate(
                code=generated_code,
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
                'semantic': aggregated.category_scores.semantic,
            }
            benchmark_result.evaluation_summary = {
                'total_evaluators': aggregated.total_evaluators,
                'passed_evaluators': aggregated.passed_evaluators,
                'failed_evaluators': aggregated.failed_evaluators,
                'all_gates_passed': aggregated.all_gates_passed,
                'gates_passed': aggregated.gates_passed,
                'gates_total': aggregated.gates_total,
            }

            # Store detailed evaluation results
            benchmark_result.evaluation_details = [
                {
                    'name': r.evaluator_name,
                    'type': r.evaluator_type.value,
                    'method': r.evaluation_method,
                    'passed': r.passed,
                    'score': r.quality_score or r.normalized_score,
                    'raw_value': r.raw_value,
                    'confidence': r.confidence,
                    'feedback': r.feedback,
                }
                for r in eval_results
            ]

            print(f"✅ Evaluation complete: Score={aggregated.composite_score:.1f}, Tier={aggregated.overall_tier}")

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            benchmark_result.composite_score = 0.0
            benchmark_result.tier = "FAIL"
            benchmark_result.evaluation_summary = {'error': str(e)}

    async def _create_evaluation_report(
        self,
        results: List[BenchmarkResult],
        summary: Dict[str, Any],
        updater: TaskUpdater
    ) -> None:
        """Create a comprehensive evaluation report.

        Args:
            results: All benchmark results
            summary: Summary statistics
            updater: TaskUpdater for creating artifacts
        """
        report_lines = [
            "=" * 80,
            "EVALUATION REPORT",
            "=" * 80,
            "",
            f"Total Problems: {summary['total']}",
            f"Successful Executions: {summary['success_count']}",
            f"Failed Executions: {summary['failure_count']}",
            f"Average Execution Time: {summary['avg_time_sec']:.2f}s",
            "",
            f"Average Composite Score: {summary['avg_composite_score']:.1f}/100",
            "",
            "Tier Distribution:",
            f"  🥇 GOLD:   {summary['tier_distribution']['GOLD']} ({summary['tier_distribution']['GOLD']/summary['total']*100:.1f}%)",
            f"  🥈 SILVER: {summary['tier_distribution']['SILVER']} ({summary['tier_distribution']['SILVER']/summary['total']*100:.1f}%)",
            f"  🥉 BRONZE: {summary['tier_distribution']['BRONZE']} ({summary['tier_distribution']['BRONZE']/summary['total']*100:.1f}%)",
            f"  ❌ FAIL:   {summary['tier_distribution']['FAIL']} ({summary['tier_distribution']['FAIL']/summary['total']*100:.1f}%)",
            "",
            "=" * 80,
            "PER-PROBLEM RESULTS",
            "=" * 80,
            "",
        ]

        for r in results:
            tier_emoji = {
                'GOLD': '🥇',
                'SILVER': '🥈',
                'BRONZE': '🥉',
                'FAIL': '❌'
            }.get(r.tier or 'FAIL', '❓')

            report_lines.append(f"{tier_emoji} {r.problem_name} (Score: {r.composite_score:.1f}/100)")
            if r.category_scores:
                report_lines.append(f"   Correctness: {r.category_scores['correctness']:.1f}, "
                                  f"Performance: {r.category_scores['performance']:.1f}, "
                                  f"Code Quality: {r.category_scores['code_quality']:.1f}")
            report_lines.append("")

        report_text = "\n".join(report_lines)

        # Save as artifact
        await updater.add_artifact(
            name="evaluation_report.txt",
            parts=[TextPart(text=report_text)],
        )

        # Also save detailed JSON
        detailed_report = {
            'summary': summary,
            'per_problem_scores': [
                {
                    'problem_name': r.problem_name,
                    'problem_id': r.problem_id,
                    'tier': r.tier,
                    'composite_score': r.composite_score,
                    'category_scores': r.category_scores,
                    'evaluation_summary': r.evaluation_summary,
                }
                for r in results
            ]
        }

        await updater.add_artifact(
            name="evaluation_detailed_report.json",
            parts=[TextPart(text=json.dumps(detailed_report, indent=2))],
        )
