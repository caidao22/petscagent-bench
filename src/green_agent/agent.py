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
from a2a.utils import get_message_text, new_agent_text_message, get_text_parts
from src.util.a2a_comm import send_message
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import dotenv

dotenv.load_dotenv()

from petsc_compile_run_mcp_client import PetscCompileRunMCPClient


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


class Agent:
    def __init__(self, purple_agent_url, mcp_server_url, max_num_prob=None):
        self.purple_agent_url = purple_agent_url
        self.mcp_client = PetscCompileRunMCPClient(mcp_server_url)
        self.max_num_prob = max_num_prob
        self.metrics = {}
        # Initialize state here

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
                assert isinstance(
                    res_result, Message
                )  # though, a robust design should also support Task
                text_parts = get_text_parts(res_result.parts)
                assert (
                    len(text_parts) == 1
                ), "Expecting exactly one text part from the purple agent"
                purple_text = text_parts[0]
                # br.purple_agent_text = purple_text

                # Ensure output folder exists
                out_path = Path("./generated_codes")
                out_path.mkdir(exist_ok=True)

                # Parse response to find code
                # print(f"@@@ purple agent response:\n{purple_text}")
                _PATTERN = re.compile(
                    r"^Code generation successful[^\n]*\n"
                    r"file_name:\s*(?P<file_name>[^\n]+)\n"
                    r"cli_args:\s*(?P<cli_args>[^\n]+)\n"
                    r"```c\n(?P<generated_code>.*?)\n```",
                    re.DOTALL,
                )
                m = _PATTERN.search(purple_text)
                print("match:", m)
                if not m:
                    raise ValueError(
                        "Could not parse purple agent response. Probably failed to generate the code."
                    )
                file_name = m.group("file_name")
                cli_args = m.group("cli_args")
                generated_code = m.group("generated_code")
                # Save to local file
                local_file = out_path / file_name
                local_file.write_text(generated_code)
                print(f"@@@ Green agent: Saved generated code to {local_file}")

                # (2) build and run
                print(
                    f"@@@ Green agent: Compile and run the code generated by purple agent..."
                )
                await self.mcp_client.initialize()
                await self.mcp_client.upload_file(filename=local_file)
                await self.mcp_client.make(executable=pname)
                (result, response) = await self.mcp_client.run_executable(
                    executable=pname, args=cli_args
                )
                # br.file_path = local_file
                # br.cli_args = cli_args
                print(result)  # use LLM as a judge
                br.is_error = response.isError
                br.success = not br.is_error
            except Exception as e:
                br.is_error = True
                br.success = False
                br.error_message = f"{type(e).__name__}: {e}"
            finally:
                await self.mcp_client.finalize()
                br.time_used_sec = time.time() - timestamp_started
                results.append(br)
                # Update rolling summary
                summary["total"] += 1
                if br.success:
                    summary["success_count"] += 1
                else:
                    summary["failure_count"] += 1
                # Optional: per-case artifact (useful for debugging)
                await updater.add_artifact(
                    name=f"benchmark_result_{pname}.json",
                    parts=[TextPart(text=json.dumps(asdict(br), indent=2))],
                )
        # Final summary artifact
        times = [r.time_used_sec for r in results]
        summary["avg_time_sec"] = (sum(times) / len(times)) if times else None
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
        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(
                f"Done. {summary['success_count']}/{summary['total']} succeeded."
            ),
        )
