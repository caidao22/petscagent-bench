import argparse
import uvicorn
import tomllib
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from .executor import GreenAgentExecutor


def load_agent_card_toml(agent_name):
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


def start_green_agent(agent_name="green_agent", host="localhost", port=9001):
    """
    Start the Green Agent A2A HTTP server.

    This function loads the agent card configuration from TOML, fills in required
    card fields (e.g., the public URL), wires up the default A2A request handler
    with a `GreenAgentExecutor` and an in-memory task store, builds the Starlette
    application, and runs it via Uvicorn.

    Args:
        agent_name (str): Name of the agent whose card/config should be loaded.
        host (str): Interface to bind the HTTP server to.
        port (int): TCP port to bind the HTTP server to.

    Side Effects:
        Starts a blocking Uvicorn server process serving the A2A endpoints.
    """
    print("Starting green agent...")

    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url  # complete all required card fields
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
    start_green_agent()
