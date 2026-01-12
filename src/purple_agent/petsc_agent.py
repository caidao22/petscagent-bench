"""purple agent implementation - the target agent being tested."""

import uvicorn
import dotenv
import subprocess
import os
import shutil
import json
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion


dotenv.load_dotenv()

SYSTEM_CODE_CONTRACT = (
    "You are a code-generation agent.\n"
    "You MUST output a single valid C source file.\n\n"
    "Output contract (must follow exactly):\n"
    "- Output JSON with 'code' (the full valid C code) and 'cli_args' (command line arguments, e.g. '-ts_type rosw -ts_monitor')\n"
    "- Any explanation MUST be inside a C block comment /* ... */\n"
    "- Do NOT use Markdown, backticks, LaTeX, or plain text outside comments\n"
    "- The first non-comment line must be valid C\n"
    "- The output must compile with a C compiler\n"
    "- Violating this contract is a hard failure\n"
)


def prepare_purple_agent_card(url):
    skill = AgentSkill(
        id="task_fulfillment",
        name="Task Fulfillment",
        description="Handles user requests and completes tasks",
        tags=["general"],
        examples=[],
    )
    card = AgentCard(
        name="file_agent",
        description="Test agent from file",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class PetscAgentExecutor(AgentExecutor):
    def __init__(self):
        self.ctx_id_to_messages = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = [
                {
                    "role": "system",
                    "content": SYSTEM_CODE_CONTRACT,
                }
            ]
        messages = self.ctx_id_to_messages[context.context_id]
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # 1. Generate PETSc code
        response = completion(
            messages=messages,
            model="gemini/gemini-3-flash-preview",  # Switched to Gemini model
            custom_llm_provider="gemini",  # Switched to Gemini provider
            temperature=0.0,
        )
        content = response.choices[0].message.model_dump()["content"]
        # remove ```json ``` wrapper that OpenAI includes in its response
        content = content.replace('```json\n','').replace('```','')
        obj = json.loads(content)
        generated_code = obj["code"]
        cli_args = obj["cli_args"]
        # messages.append(
        #     {
        #         "role": "assistant",
        #         "content": generated_code,
        #     }
        # )

        # Create a unique working directory for this context
        work_dir = "generated_codes"
        os.makedirs(work_dir, exist_ok=True)

        try:
            # 2. Save the generated code to a file inside the work_dir
            source_filename = os.path.join(work_dir, f"{context.context_id}.c")
            with open(source_filename, "w") as f:
                f.write(generated_code)

            # 3. Copy the template makefile to the work_dir
            # Note: The makefile should be in a known location relative to the script.
            # Assuming it's in the same directory as this script.
            # current_dir = os.path.dirname(__file__)
            # template_makefile_path = os.path.join(current_dir, "makefile")
            # shutil.copy(template_makefile_path, os.path.join(work_dir, "makefile"))

            # 4. Compile the code using the makefile in the work_dir
            # compile_result = subprocess.run(
            #     ["make", "generated_code"], capture_output=True, text=True, cwd=work_dir
            # )
            # if compile_result.returncode != 0:
            #     await event_queue.enqueue_event(
            #         new_agent_text_message(
            #             f"Compilation failed:\n{compile_result.stderr}",
            #             context_id=context.context_id,
            #         )
            #     )
            #     return

            # 5. Run the executable
            # executable_path = "./generated_code"
            # run_result = subprocess.run(
            #     [executable_path, "-nox"], capture_output=True, text=True, cwd=work_dir
            # )
            # if run_result.returncode != 0:
            #     await event_queue.enqueue_event(
            #         new_agent_text_message(
            #             f"Execution failed:\n{run_result.stderr}",
            #             context_id=context.context_id,
            #         )
            #     )
            #     return

            # 6. Return the output
            await event_queue.enqueue_event(
                new_agent_text_message(
                    # f"Execution successful:\n{run_result.stdout}",
                    f"Execution successful:\n{source_filename}\n{cli_args}",
                    context_id=context.context_id,
                )
            )

        finally:
            # 7. Clean up the working directory
            # shutil.rmtree(work_dir)
            pass

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_purple_agent(agent_name="petsc_agent", host="localhost", port=9002):
    print("Starting petsc agent...")
    url = f"http://{host}:{port}"
    card = prepare_purple_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=PetscAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    start_purple_agent()
