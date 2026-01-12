import argparse
import uvicorn
import tomllib
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from .executor import GreenAgentExecutor
from loguru import logger


def load_agent_card_toml(agent_name):
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)

def start_green_agent(host: str="localhost", port: int=9001, card_url: str=None):
    """Start the Green Agent A2A HTTP server.

    Loads the agent card configuration from ``green_agent.toml``, ensures the
    required ``url`` field is populated (from ``card_url`` or
    ``http://{host}:{port}``), wires up a :class:`~a2a.server.request_handlers.DefaultRequestHandler`
    with :class:`~petscagent_bench.src.green_agent.executor.GreenAgentExecutor`
    and an :class:`~a2a.server.tasks.InMemoryTaskStore`, then starts a Starlette
    app via Uvicorn.

    Args:
        host: Interface to bind the server to.
        port: Port to bind the server to.
        card_url: External URL to advertise in the agent card. If empty,
            ``http://{host}:{port}`` is used.

    Returns:
        None
    """
    logger.info("Starting green agent...")
    agent_card_dict = load_agent_card_toml("green_agent")
    agent_card_dict["url"] = card_url or f"http://{host}:{port}" # complete all required card fields
    request_handler = DefaultRequestHandler(
        agent_executor=GreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the green agent.")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9001, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    # parser.add_argument("--agent-llm", type=str, default="openai/gpt-4.1", help="LLM model to use")
    args = parser.parse_args()

    start_green_agent(args.host, args.port, args.card_url)