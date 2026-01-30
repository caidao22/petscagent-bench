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
import json
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_parts_message
from a2a.types import TextPart, FilePart, FileWithBytes
import litellm
from litellm import completion
from pydantic import BaseModel
from loguru import logger
from typing import Any, Dict
from pathlib import Path

dotenv.load_dotenv()

# System prompt that defines the code generation contract
# This ensures the LLM produces output in a structured, parseable format
SYSTEM_CODE_CONTRACT = (
    "You are a code-generation agent.\n"
    "Return ONLY a single raw JSON object. No markdown, no backticks, no code blocks, no explanation outside the JSON.\n"
    "Top-level JSON keys MUST be exactly: 'codes', 'nsize', 'cli_args' (no additional keys).\n\n"
    "Rules:\n"
    "- 'codes': a list of objects with 'filename' and 'code'. Code must be valid C/C++/CUDA.\n"
    "- 'nsize': the number of MPI processes (use 1 for sequential).\n"
    "- 'cli_args': command line arguments string.\n"
    "- First file in 'codes' is the main file.\n"
    "- Any explanations MUST be inside C block comments /* ... */ within the code strings.\n"
)

def load_purple_agent_config(config_path: str = "config/purple_agent_config.yaml") -> Dict[str, Any]:
    """Load purple agent configuration from file or use defaults.

    Supports both JSON and YAML formats. Format is auto-detected by file extension.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                # Detect format by extension
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            print(f"@@@ Purple agent: ✅ Loaded agent config from {config_path}")
            return config_data
        except Exception as e:
            print(f"@@@ Purple agent: ⚠️ Failed to load config from {config_path}: {e}")
            print(f"@@@ Purple agent: Using default evaluation configuration")
    else:
        print(f"@@@ Purple agent: Config file {config_path} not found, using defaults")

    # Fall back to default configuration
    return {
        'llm': {
            'model': 'gemini/gemini-3-flash-preview',
            'api_base': None,
            'temperature': 0.3,
            'max_concurrent_calls': 3,
        },
    }

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

    def __init__(self, model: str, api_base_url: str | None = None):
        """Initialize the executor with a specific LLM model.

        Args:
            model: LLM model string compatible with litellm
                   (e.g., "openai/gpt-4o", "gemini/gemini-2.5-flash")
            api_base_url: Optional API base URL (e.g., "https://api.openai.com/v1").
                         If None, uses LiteLLM default based on model provider.
        """
        self.model = model
        self.api_base_url = api_base_url
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

        class CodeFile(BaseModel):
            filename: str
            code: str

        class ProblemResponse(BaseModel):
            codes: list[CodeFile]
            nsize: int    # e.g. 1 for sequential codes
            cli_args: str # e.g. "-ts_type beuler -ts_monitor"

        try:
            # Generate PETSc code using the LLM with JSON schema
            completion_kwargs = {
                'messages': messages,
                'model': self.model,
                'temperature': 0.0,
                'response_format': ProblemResponse
            }
            litellm.ssl_verify = False
            if self.api_base_url:
                completion_kwargs['api_base'] = self.api_base_url
                is_asksage_endpoint = self.api_base_url.startswith('https://api.asksage.anl.gov')
                if is_asksage_endpoint:
                    litellm.ssl_verify = os.environ["ASKSAGE_SSL_CERT_FILE"]
                    completion_kwargs['api_key'] = os.environ["ASKSAGE_API_KEY"]
            response = completion(**completion_kwargs)
            # Extract the generated content from LLM response
            content = response.choices[0].message.content
            if isinstance(content, str):
                # Remove markdown code block wrapper that some LLMs such as Claude add
                # This ensures we get clean JSON for parsing
                if content.startswith("```"):
                    content = content.split("```", 2)[1]
                    content = content.lstrip("json").strip()
                data = json.loads(content)
            else:
                raise TypeError(f"Expected string content from response, got {type(content)}")
            # Parse the JSON response
            nsize = data["nsize"]
            cli_args = data["cli_args"]
            parts_list = [TextPart(text=f"Code generation successful ✅\nnsize: {nsize}\ncli_args: {cli_args}\n")]
            for entry in data["codes"]:
                # Create file object with code content
                fwb = FileWithBytes(
                    name=entry["filename"],
                    bytes=entry["code"].encode("utf-8"),
                    mime_type="text/plain"
                )
                parts_list.append(FilePart(file=fwb))
            # Send the successful response back to the client
            await event_queue.enqueue_event(
                new_agent_parts_message(parts_list, context_id=context.context_id)
            )
        except Exception as e:
            # Handle any errors during code generation
            print(f"@@@ Purple agent: ❌ Task failed with agent error: {e}")
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

def start_purple_agent(host: str = "localhost", port: int=9002, card_url: str | None = None, agent_llm: str = "gemini/gemini-3-flash-preview", api_base_url: str | None = None):
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
        api_base_url: Optional API base URL for LLM API calls.

    Notes:
        Although declared `async`, this function starts a blocking Uvicorn server
        (`uvicorn.run(...)`) and will not return until the server stops.

    """
    logger.info("Starting purple agent...")
    card = prepare_purple_agent_card(card_url or f"http://{host}:{port}")

    request_handler = DefaultRequestHandler(
        agent_executor=PetscAgentExecutor(agent_llm, api_base_url=api_base_url),
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
    parser.add_argument("--api-base-url", type=str, help="Optional LLM API base URL (overrides config)")
    args = parser.parse_args()

    start_purple_agent(args.host, args.port, args.card_url, args.agent_llm, api_base_url=args.api_base_url)
