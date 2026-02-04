"""Coding agent experiment: A2A code-generation agent that uses the same MCP tools as the green agent.

This agent is a variant of the purple agent with MCP enabled. It connects to the
same PETSc compile-run MCP server the green agent uses, so it can create files,
compile, and run code during generation (e.g. to validate or iterate).

Architecture Overview:
----------------------
1. This agent runs as an A2A (Agent-to-Agent) HTTP server using the A2A protocol.
2. It receives code generation requests from clients (e.g., an orchestrator agent).
3. It uses an LLM (via litellm) to generate PETSc/C/C++/CUDA code.
4. The LLM has access to MCP tools (create_file, compile, run) to validate code.
5. The agent returns the generated code as JSON with 'codes', 'nsize', 'cli_args'.

Key Components:
---------------
- CodingAgentExecutor: The main executor that handles code generation requests.
- _run_with_mcp(): Agentic loop that allows the LLM to use MCP tools iteratively.
- start_agent(): Entry point to launch the A2A HTTP server.
"""

import multiprocessing
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
import litellm
import uvicorn
from litellm import completion
from loguru import logger

# A2A (Agent-to-Agent) protocol imports for building the agent server
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.types import FilePart, FileWithBytes, TextPart
from a2a.utils import new_agent_parts_message

# Load environment variables (API keys, etc.)
dotenv.load_dotenv()

# MCP (Model Context Protocol) imports for tool use
# - start_mcp_server: Function to launch the PETSc compile-run MCP server
# - PetscCompileRunMCPClient: Client to communicate with the MCP server
# - mcpdynamicclient: Contains exception classes for MCP tool errors
from petsc_compile_run_mcp_server import main as start_mcp_server
from petsc_compile_run_mcp_client import PetscCompileRunMCPClient, mcpdynamicclient


# =============================================================================
# System Prompts
# =============================================================================
# These prompts instruct the LLM on the expected output format.

# Base contract: LLM must return raw JSON with specific keys
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

# Extended contract: Adds MCP tool usage instructions
SYSTEM_CODE_CONTRACT_WITH_MCP = (
    SYSTEM_CODE_CONTRACT
    + "\n\n"
    + "You have access to MCP tools (same PETSc compile-run environment as the evaluator). "
    "You MAY call these tools to create files, compile, or run code to validate your solution before returning the final JSON. "
    "When you are done (or if you skip tool use), return the single JSON object with 'codes', 'nsize', 'cli_args' as above.\n"
)


# =============================================================================
# Helper Functions for MCP Tool Integration
# =============================================================================

def _mcp_tools_to_openai(mcp_tool_list: List[Any]) -> List[Dict[str, Any]]:
    """Convert MCP tool definitions to OpenAI/litellm function calling format.
    
    MCP tools have their own schema format. This function converts them to the
    format expected by OpenAI's function calling API (used by litellm).
    
    Args:
        mcp_tool_list: List of MCP tool objects from session.list_tools()
        
    Returns:
        List of tool definitions in OpenAI format, ready for the 'tools' parameter
    """
    openai_tools = []
    for t in mcp_tool_list:
        name = getattr(t, "name", None)
        if not name:
            continue
        desc = getattr(t, "description", None) or ""
        # inputSchema defines the parameters the tool accepts
        schema = getattr(t, "inputSchema", None)
        if schema is None:
            schema = {"type": "object", "properties": {}}
        openai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": schema,
            },
        })
    return openai_tools


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize tool_calls from LLM response to a consistent dict format.
    
    Different LLM providers return tool calls in slightly different formats.
    This function normalizes them to a consistent dict structure that can be
    added to the message history for the next API call.
    
    Args:
        tool_calls: Tool calls from the LLM response (may be objects or dicts)
        
    Returns:
        List of normalized tool call dicts with 'id', 'type', and 'function' keys
    """
    if not tool_calls:
        return []
    out = []
    for tc in tool_calls:
        # Already a dict - use as-is
        if isinstance(tc, dict):
            out.append(tc)
            continue
        # Extract function info from object attributes
        f = getattr(tc, "function", None)
        if isinstance(f, dict):
            name = f.get("name", "")
            args = f.get("arguments", "{}")
        else:
            name = getattr(f, "name", "") if f else ""
            args = getattr(f, "arguments", "{}") if f else "{}"
        out.append({
            "id": getattr(tc, "id", ""),
            "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
        })
    return out


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str = "config/coding_agent_exp_config.yaml") -> Dict[str, Any]:
    """Load agent configuration from a YAML or JSON file.
    
    The config file specifies:
    - LLM settings: model name, temperature, API base URL
    - MCP settings: server URL, enabled flag
    
    If the file doesn't exist or fails to load, returns sensible defaults.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dict with 'llm' and 'mcp' sections
    """
    config_file = Path(config_path)
    
    # Default configurations
    default_mcp = {"server_url": "http://localhost:8080/mcp"}
    default_llm = {
        "model": "gemini/gemini-3-flash-preview",
        "api_base_url": None,
        "temperature": 0.3,
    }
    
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                import yaml
                config_data = yaml.safe_load(f)
            # Ensure required sections exist
            if "mcp" not in config_data:
                config_data["mcp"] = default_mcp
            if "llm" not in config_data:
                config_data["llm"] = default_llm
            logger.info(f"Coding agent exp: loaded config from {config_path}")
            return config_data
        except Exception as e:
            logger.warning(f"Coding agent exp: failed to load {config_path}: {e}")
    
    # Return defaults if config file missing or failed to load
    return {
        "llm": default_llm,
        "mcp": default_mcp,
    }


# =============================================================================
# A2A Agent Card
# =============================================================================

def prepare_agent_card(url: str) -> AgentCard:
    """Create an A2A AgentCard that describes this agent's capabilities.
    
    The AgentCard is served at /.well-known/agent.json and tells other agents:
    - What this agent does (description)
    - What skills it has
    - What input/output formats it supports
    
    Args:
        url: The base URL where this agent will be accessible
        
    Returns:
        AgentCard object describing this agent
    """
    skill = AgentSkill(
        id="task_fulfillment",
        name="Task Fulfillment",
        description="Handles user requests and completes tasks (with MCP tools)",
        tags=["general"],
        examples=[],
    )
    return AgentCard(
        name="coding_agent_exp",
        description="Code generation agent with MCP tools (same as green agent)",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


# =============================================================================
# Main Agent Executor
# =============================================================================

class CodingAgentExecutor(AgentExecutor):
    """Executor that generates PETSc code using MCP tools for validation.
    
    This executor implements the A2A AgentExecutor interface. When a request
    comes in, it:
    1. Adds the user's message to the conversation history
    2. Calls the LLM with MCP tools available
    3. If the LLM uses tools, executes them and continues the loop
    4. When the LLM returns final JSON, parses and returns it
    
    The MCP tools allow the LLM to:
    - Create files in a sandbox environment
    - Compile PETSc/C/C++ code
    - Run the compiled code and see output
    This enables iterative debugging before returning the final answer.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the executor with configuration.
        
        Args:
            config: Configuration dict with 'llm' and 'mcp' sections
        """
        self.config = config
        
        # LLM configuration
        llm = config.get("llm", {})
        self.model = llm.get("model")
        self.temperature = float(llm.get("temperature", 0.3))
        self.api_base_url = llm.get("api_base_url")

        # MCP server configuration
        mcp_cfg = config.get("mcp") or {}
        self.mcp_server_url = (mcp_cfg.get("server_url") or "http://localhost:8080/mcp").strip()

        # Conversation history per context (for multi-turn conversations)
        self.ctx_id_to_messages: Dict[str, List[Dict[str, Any]]] = {}

    async def _run_with_mcp(
        self,
        messages: List[Dict[str, Any]],
        completion_kwargs_base: Dict[str, Any],
    ) -> tuple[Optional[Dict], Optional[str]]:
        """Run the agentic loop with MCP tools until the LLM returns final JSON.
        
        This is the core agentic loop:
        1. Send messages + tools to LLM
        2. If LLM returns tool calls, execute them and add results to messages
        3. Repeat until LLM returns content without tool calls
        4. Parse the final content as JSON and return
        
        Args:
            messages: Conversation history (system + user messages)
            completion_kwargs_base: Base kwargs for litellm.completion()
            
        Returns:
            Tuple of (parsed_data, error_message). One will be None.
        """
        # Create MCP client and connect to server
        client = PetscCompileRunMCPClient(self.mcp_server_url)
        try:
            await client.initialize()
        except Exception as e:
            return None, f"MCP client failed to connect to {self.mcp_server_url}: {e}"

        try:
            # Get available tools from MCP server
            # list_tools() returns an iterable of tool objects
            tools_result = await client.session.list_tools()
            openai_tools = _mcp_tools_to_openai(list(tools_result))
            if not openai_tools:
                return None, "MCP server returned no tools"
            
            # Log available tools for debugging
            tool_names = [t["function"]["name"] for t in openai_tools]
            logger.info(f"Available MCP tools: {tool_names}")

            # Create a copy of messages for the loop (don't modify original)
            loop_messages = [m for m in messages]
            
            # Ensure system prompt includes MCP instructions
            if loop_messages and loop_messages[0].get("role") == "system":
                loop_messages[0] = {"role": "system", "content": SYSTEM_CODE_CONTRACT_WITH_MCP}
            else:
                loop_messages.insert(0, {"role": "system", "content": SYSTEM_CODE_CONTRACT_WITH_MCP})

            # Agentic loop: allow up to N rounds of tool use
            max_tool_rounds = 15
            for _ in range(max_tool_rounds):
                # Call LLM with current messages and available tools
                completion_kwargs = {
                    **completion_kwargs_base,
                    "messages": loop_messages,
                    "tools": openai_tools,
                }
                completion_kwargs.pop("response_format", None)  # Not compatible with tools
                response = completion(**completion_kwargs)
                msg = response.choices[0].message
                
                # Add assistant's response to message history
                raw_tool_calls = getattr(msg, "tool_calls", None) or []
                loop_messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": _normalize_tool_calls(raw_tool_calls),
                })

                # Check if LLM wants to use tools
                tool_calls = getattr(msg, "tool_calls", None) or []
                if not tool_calls:
                    # No tool calls = LLM is done, parse the final response
                    content = (msg.content or "").strip()
                    if not content:
                        return None, "Model returned empty content"
                    
                    # Strip markdown code blocks if present
                    if content.startswith("```"):
                        content = content.split("```", 2)[1]
                        content = content.lstrip("json").strip()
                    
                    # Parse and validate JSON
                    try:
                        data = json.loads(content)
                        if "codes" in data and "nsize" in data and "cli_args" in data:
                            return data, None
                    except json.JSONDecodeError as e:
                        return None, f"Invalid JSON in final response: {e}"
                    return None, "Final response missing 'codes', 'nsize', or 'cli_args'"

                # Execute each tool call and add results to messages
                for tc in tool_calls:
                    tid = getattr(tc, "id", None) or ""
                    f = getattr(tc, "function", None) or {}
                    name = f.get("name") if isinstance(f, dict) else getattr(f, "name", "")
                    args_str = f.get("arguments", "{}") if isinstance(f, dict) else getattr(f, "arguments", "{}")
                    
                    # Parse tool arguments
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                    except json.JSONDecodeError:
                        args = {}
                    
                    logger.info(f"Calling MCP tool: {name} with args: {args}")
                    
                    # Call the MCP tool and capture result/error
                    try:
                        result = await client.call_tool(name, args)
                        result_text = json.dumps(result) if isinstance(result, dict) else str(result)
                        logger.info(f"Tool {name} result: {result_text[:200]}..." if len(result_text) > 200 else f"Tool {name} result: {result_text}")
                    except mcpdynamicclient.MCPDynamicClientReturnCode as e:
                        # Tool returned non-zero exit code (e.g., compilation failed)
                        result_text = json.dumps({
                            "error": str(e),
                            "stdout": getattr(e, "stdout", ""),
                            "stderr": getattr(e, "stderr", ""),
                            "returncode": getattr(e, "returncode", None),
                        })
                    except mcpdynamicclient.MCPDynamicClientToolError as e:
                        result_text = json.dumps({"error": str(e)})
                    except Exception as e:
                        result_text = json.dumps({"error": str(e)})
                    
                    # Add tool result to messages for next LLM call
                    loop_messages.append({"role": "tool", "tool_call_id": tid, "content": result_text})

            return None, f"Exceeded max tool rounds ({max_tool_rounds})"
        finally:
            # Always clean up the MCP client connection
            try:
                await client.finalize()
            except Exception:
                pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle an incoming A2A request.
        
        This is the main entry point called by the A2A framework when a request
        comes in. It orchestrates the code generation and sends the result back
        through the event queue.
        
        Args:
            context: Request context containing user input and context ID
            event_queue: Queue for sending response events back to the client
        """
        # Get user input from the request
        user_input = context.get_user_input()
        
        # Initialize or retrieve conversation history for this context
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = [
                {"role": "system", "content": SYSTEM_CODE_CONTRACT_WITH_MCP},
            ]
        messages = self.ctx_id_to_messages[context.context_id]
        messages.append({"role": "user", "content": user_input})

        # Prepare base kwargs for LLM calls
        completion_kwargs_base = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": 300,
        }
        
        # Configure SSL and API settings
        litellm.ssl_verify = False
        if self.api_base_url:
            completion_kwargs_base["api_base"] = self.api_base_url
            # Special handling for ANL's AskSage API
            if self.api_base_url.startswith("https://api.asksage.anl.gov"):
                litellm.ssl_verify = os.environ.get("ASKSAGE_SSL_CERT_FILE", True)
                completion_kwargs_base["api_key"] = os.environ.get("ASKSAGE_API_KEY", "")

        try:
            # Run the agentic loop with MCP tools
            data, err = await self._run_with_mcp(messages, completion_kwargs_base)
            if err is not None:
                raise RuntimeError(err)

            # Build success response with generated code files
            nsize = data["nsize"]
            cli_args = data["cli_args"]
            parts_list = [
                TextPart(text=f"Code generation successful ✅\nnsize: {nsize}\ncli_args: {cli_args}\n")
            ]
            # Add each generated file as a FilePart
            for entry in data["codes"]:
                parts_list.append(
                    FilePart(
                        file=FileWithBytes(
                            name=entry["filename"],
                            bytes=entry["code"].encode("utf-8"),
                            mime_type="text/plain",
                        )
                    )
            )
            # Send response back through event queue
            await event_queue.enqueue_event(
                new_agent_parts_message(parts_list, context_id=context.context_id)
            )
        except Exception as e:
            # Log error and send failure response
            logger.exception("Coding agent exp: task failed")
            await event_queue.enqueue_event(
                new_agent_parts_message(
                    [TextPart(text=f"Code generation failed ❌\nerror: {e}\n")],
                    context_id=context.context_id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation of a running request.
        
        Currently not implemented - would need to track running tasks and
        provide a way to interrupt them.
        """
        raise NotImplementedError


# =============================================================================
# Agent Server Entry Point
# =============================================================================

def start_agent(
    host: str = "localhost",
    port: int = 9003,
    card_url: Optional[str] = None,
    agent_llm: Optional[str] = None,
    api_base_url: Optional[str] = None,
    config_path: str = "config/coding_agent_exp_config.yaml",
    mcp_server_url: Optional[str] = None,
) -> None:
    """Start the coding agent as an A2A HTTP server.
    
    This function:
    1. Loads configuration from file
    2. Applies any command-line overrides
    3. Creates the A2A application with our executor
    4. Starts the uvicorn HTTP server
    
    Args:
        host: Host to bind the server to
        port: Port to listen on (default 9003 to avoid conflicts)
        card_url: Override URL in the agent card (for reverse proxies)
        agent_llm: Override the LLM model to use
        api_base_url: Override the API base URL for the LLM
        config_path: Path to the configuration file
        mcp_server_url: Override the MCP server URL
    """
    logger.info("Starting coding agent exp...")
    
    # Create agent card with the server URL
    card = prepare_agent_card(card_url or f"http://{host}:{port}")
    
    # Load config and apply overrides
    config = load_config(config_path)
    if agent_llm:
        config["llm"]["model"] = agent_llm
    if api_base_url:
        config["llm"]["api_base_url"] = api_base_url
    if mcp_server_url:
        config["mcp"]["server_url"] = mcp_server_url

    # Create the A2A request handler with our executor
    request_handler = DefaultRequestHandler(
        agent_executor=CodingAgentExecutor(config),
        task_store=InMemoryTaskStore(),
    )
    
    # Build and run the A2A application
    app = A2AStarletteApplication(agent_card=card, http_handler=request_handler)
    uvicorn.run(app.build(), host=host, port=port)


# =============================================================================
# Test Mode
# =============================================================================

async def run_test(agent_url: str, test_prompt: str) -> None:
    """Send a test request to the agent and print the response.
    
    Args:
        agent_url: URL of the running agent (e.g., http://localhost:9003)
        test_prompt: The code generation prompt to send
    """
    from src.util.a2a_comm import wait_agent_ready, send_message
    
    assert await wait_agent_ready(agent_url), "Agent not ready in time"
    print("Agent is ready.")
    
    response = await send_message(agent_url, test_prompt)
    
    print("\n[Test] Response:")
    print("-" * 60)
    print(response)
    print("-" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the coding agent experiment (MCP-enabled).")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=9003,
                        help="Port to listen on (default 9003)")
    parser.add_argument("--card-url", type=str,
                        help="Override URL in agent card (for reverse proxies)")
    parser.add_argument("--config", type=str, default="config/coding_agent_exp_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--api-base-url", type=str,
                        help="Override API base URL for LLM")
    parser.add_argument("--agent-llm", type=str,
                        help="Override LLM model to use")
    parser.add_argument("--mcp-server-url", type=str, default=None,
                        help="Override MCP server URL")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode: start server, send test request, then exit")
    args = parser.parse_args()

    # Start MCP server first (the agent depends on it for tool execution)
    print("Launching MCP server for coding agent...")
    petsc_mcp_server = multiprocessing.Process(target=start_mcp_server)
    petsc_mcp_server.start()
    time.sleep(2)  # Give MCP server time to initialize
    print("PETSc MCP server started.")

    agent_url = f"http://{args.host}:{args.port}"

    if args.test:
        # Test mode: start agent in subprocess, send test request, then exit
        def run_agent_process():
            start_agent(
                host=args.host,
                port=args.port,
                card_url=args.card_url,
                agent_llm=args.agent_llm,
                api_base_url=args.api_base_url,
                config_path=args.config,
                mcp_server_url=args.mcp_server_url,
            )
        
        print("Launching coding agent (test mode)...")
        agent_process = multiprocessing.Process(target=run_agent_process)
        agent_process.start()
        
        try:
            # Run the test with a prompt that encourages tool use
            test_prompt = (
                "Write a sequential PETSc program (nsize=1) that creates a vector of size 10, "
                "sets each element to its global index (0,1,2,...,9), computes the sum of all elements "
                "using VecSum, and prints the result with PetscPrintf. The expected sum is 0+1+2+...+9 = 45. "
                "Please compile and run your code to verify it produces the correct output "
                "before returning the final answer."
            )
            asyncio.run(run_test(agent_url, test_prompt))
        finally:
            # Cleanup
            print("Shutting down agent...")
            agent_process.terminate()
            agent_process.join(timeout=5)
            if agent_process.is_alive():
                agent_process.kill()
            
            print("Shutting down MCP server...")
            petsc_mcp_server.terminate()
            petsc_mcp_server.join(timeout=5)
            if petsc_mcp_server.is_alive():
                petsc_mcp_server.kill()
            print("Done.")
    else:
        # Normal mode: run agent server (blocks until Ctrl+C)
        try:
            print("Launching coding agent...")
            start_agent(
                host=args.host,
                port=args.port,
                card_url=args.card_url,
                agent_llm=args.agent_llm,
                api_base_url=args.api_base_url,
                config_path=args.config,
                mcp_server_url=args.mcp_server_url,
            )
        finally:
            # Cleanup: terminate MCP server when agent stops (Ctrl+C or error)
            print("Shutting down MCP server...")
            petsc_mcp_server.terminate()
            petsc_mcp_server.join(timeout=5)
            if petsc_mcp_server.is_alive():
                petsc_mcp_server.kill()
            print("PETSc MCP server terminated.")
