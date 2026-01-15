"""Purple Agent implementation - the target code generation agent being tested.

The Purple Agent is responsible for generating PETSc C/C++ code from natural
language problem descriptions. It operates as an A2A-compliant agent that:

1. Receives problem descriptions via the A2A protocol
2. Uses an LLM to generate PETSc code
3. Returns generated code files along with CLI arguments
4. Maintains conversation context for multi-turn interactions

The agent is isolated from the evaluation logic and is the subject of testing
by the Green Agent through the petscagent-bench framework.
"""

import argparse
import uvicorn
import dotenv
import os
# import subprocess
# import shutil
import json
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_parts_message
from a2a.types import TextPart, FilePart, FileWithBytes
from litellm import completion
from loguru import logger

dotenv.load_dotenv()

# System prompt that defines the code generation contract
# This ensures the LLM produces output in a structured, parseable format
SYSTEM_CODE_CONTRACT = (
    "You are a code-generation agent.\n"
    "You MUST output valid C/C++ source files.\n\n"
    "Output contract (must follow exactly):\n"
    "- Output JSON with 'codes' and 'cli_args'\n"
    "- 'codes' MUST be a list of objects, each with:\n"
    "    - 'filename': the name of the source file (e.g., 'main.c')\n"
    "    - 'code': the full contents of that file\n"
    "    - the first file must be the main file\n"
    "- 'cli_args' contains command line arguments (e.g., '-ts_type rosw -ts_monitor')\n"
    "- Any explanation MUST be inside a C block comment /* ... */\n"
    "- Do NOT use Markdown, backticks, LaTeX, or plain text outside comments\n"
    "- The first non-comment line must be valid C\n"
    "- The output must compile with a C/C++ compiler\n"
    "- Violating this contract is a hard failure\n"
)


def prepare_purple_agent_card(url):
    """Create an A2A agent card for the Purple Agent.
    
    The agent card is a metadata descriptor that advertises the agent's
    capabilities, skills, and communication modes to potential clients.
    
    Args:
        url: The base URL where this agent is accessible
    
    Returns:
        AgentCard object describing the Purple Agent's capabilities
    """
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
    """Executor class that handles code generation requests.
    
    This class implements the AgentExecutor interface from the A2A framework.
    It manages:
    - LLM interactions for code generation
    - Conversation context tracking
    - Response formatting and validation
    - Error handling and reporting
    
    Attributes:
        model: LLM model identifier (e.g., "openai/gpt-4o", "gemini/gemini-2.5-flash")
        ctx_id_to_messages: Dict mapping context IDs to conversation histories
    """
    
    def __init__(self, model: str):
        """Initialize the executor with a specific LLM model.
        
        Args:
            model: LLM model string compatible with litellm
                   (e.g., "openai/gpt-4o", "gemini/gemini-2.5-flash")
        """
        self.model = model
        # Track conversation history per context for multi-turn interactions
        self.ctx_id_to_messages = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a code generation request.
        
        This method is called by the A2A framework when a message is received.
        It processes the request through the following steps:
        
        1. Extract user input from the request context
        2. Initialize or retrieve conversation history
        3. Call the LLM to generate PETSc code
        4. Parse and validate the LLM response
        5. Format the response as A2A message parts (text + files)
        6. Enqueue the response event
        
        Args:
            context: Request context containing user input and metadata
            event_queue: Queue for sending response events back to the client
        
        Note:
            The method handles both success and error cases, always returning
            a properly formatted A2A response.
        """
        user_input = context.get_user_input()
        
        # Initialize conversation history for new contexts
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

        try:
            # Generate PETSc code using the LLM
            response = completion(
                messages=messages,
                model=self.model,
                # custom_llm_provider="openai",
                temperature=0.0,
            )
            # Extract the generated content from LLM response
            content = response.choices[0].message.model_dump()["content"]
            
            # Remove markdown code block wrapper that some LLMs add
            # This ensures we get clean JSON for parsing
            content = content.replace('```json\n','').replace('```','')
            
            # Parse the JSON response
            obj = json.loads(content)
            cli_args = obj["cli_args"]
            
            # Build response parts: text message + file attachments
            parts_list = [TextPart(text=f"Code generation successful ✅\ncli_args: {cli_args}\n")]
            
            # Add each generated code file as a FilePart
            for entry in obj["codes"]:
                # Create file object with code content
                fwb = FileWithBytes(
                    name=entry["filename"],
                    bytes=entry["code"].encode("utf-8"),
                    mime_type="text/plain"
                )
                parts_list.append(FilePart(file=fwb))
            
            # Rename the first code file to use the context ID for tracking
            if len(parts_list) > 1:
                first_entry = obj["codes"][0]
                ext = first_entry["filename"].split('.')[-1]
                parts_list[1].file.name = f"{context.context_id}.{ext}"
            # messages.append(
            #     {
            #         "role": "assistant",
            #         "content": generated_code,
            #     }
            # )

            # create a unique working directory for this context
            # work_dir = "generated_codes"
            # os.makedirs(work_dir, exist_ok=True)
            # 2. Save the generated code to a file inside the work_dir
            # source_filename = os.path.join(work_dir, f"{context.context_id}.c")
            # with open(source_filename, "w") as f:
            #     f.write(generated_code)
            # Send the successful response back to the client
            await event_queue.enqueue_event(
                new_agent_parts_message(parts_list, context_id=context.context_id)
            )
        except Exception as e:
            # Handle any errors during code generation
            print(f"Task failed with agent error: {e}")
            
            # Return error message to the client
            parts_list = [TextPart(text=f"Code generation failed ❌\nerror: {e}\n")]
            await event_queue.enqueue_event(
                new_agent_parts_message(parts_list, context_id=context.context_id)
            )

    async def cancel(self, context, event_queue) -> None:
        """Cancel a running task (not implemented).
        
        This method is required by the AgentExecutor interface but is not
        currently implemented as code generation tasks are atomic and fast.
        
        Args:
            context: Request context
            event_queue: Event queue for sending cancellation acknowledgment
        
        Raises:
            NotImplementedError: Always, as cancellation is not supported
        """
        raise NotImplementedError

def start_purple_agent(host: str="localhost", port: int=9002, card_url: str=None, agent_llm: str="gemini/gemini-3-flash-preview"):
    """
    Start the Purple Agent A2A HTTP service.

    This sets up an A2A Starlette application backed by a `PetscAgentExecutor`,
    an in-memory task store, and the default request handler, then serves it
    via Uvicorn.

    Args:
        host: Interface to bind the HTTP server to.
        port: Port to bind the HTTP server to.
        card_url: Optional explicit URL used to build/advertise the agent card.
            If not provided, a URL is constructed from `host` and `port`.
        agent_llm: LLM identifier/config string passed to `PetscAgentExecutor`.

    Notes:
        Although declared `async`, this function starts a blocking Uvicorn server
        (`uvicorn.run(...)`) and will not return until the server stops.

    """
    logger.info("Starting purple agent...")
    card = prepare_purple_agent_card(card_url or f"http://{host}:{port}")

    request_handler = DefaultRequestHandler(
        agent_executor=PetscAgentExecutor(agent_llm),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    uvicorn.run(app.build(), host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the purple agent.")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    parser.add_argument("--agent-llm", type=str, default="gemini/gemini-3-flash-preview", help="LLM model to use, such as gemini/gemini-3-flash-preview, gemini/gemini-2.5-flash, gpt-5.2")
    args = parser.parse_args()

    start_purple_agent(args.host, args.port, args.card_url, args.agent_llm)
