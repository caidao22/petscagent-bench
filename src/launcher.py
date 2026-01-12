import multiprocessing
import asyncio
import json
import mcp
from src.green_agent.server import start_green_agent
from src.purple_agent.petsc_agent import start_purple_agent
from src.util.a2a_comm import wait_agent_ready, send_message
import os
import dotenv
# load environment variables before importing server code
dotenv.load_dotenv()
from petsc_compile_run_mcp_server import main as start_mcp_server

def run_green_agent():
    asyncio.run(start_green_agent())

def run_purple_agent():
    asyncio.run(start_purple_agent())

async def launch_evaluation():
    """Launcher module - initiates and coordinates the evaluation process."""
    # start green agent
    green_url = "http://localhost:9001"
    purple_url = "http://localhost:9002"
    print("Launching green agent...")
    p_green = multiprocessing.Process(target=run_green_agent)
    p_green.start()
    assert await wait_agent_ready(green_url), "Green agent not ready in time"
    print("Green agent is ready.")

    # start purple agent
    print("Launching purple agent...")
    p_purple = multiprocessing.Process(target=run_purple_agent)
    p_purple.start()
    assert await wait_agent_ready(purple_url), "purple agent not ready in time"
    print("purple agent is ready.")

     # start the MCP server for green agent
    print("Launching MCP server for green agent...")
    petsc_mcp_server = multiprocessing.Process(target=start_mcp_server)
    petsc_mcp_server.start()
    print("PETSc MCP server is ready.")

    # send the task description
    print("Sending task description to green agent...")
    task_text = f"""
Your task is to instantiate petscagent-bench to test the agent located at:
<purple_agent_url>
{purple_url}/
</purple_agent_url>
You should use the following env configuration:
    """
    print("Task description:")
    print(task_text)
    print("Sending...")
    response = await send_message(green_url, task_text)
    print("Response from green agent:")
    print(response)

    print("Evaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_purple.terminate()
    p_purple.join()
    print("Agents terminated.")
    petsc_mcp_server.terminate()
    petsc_mcp_server.join()
    print("PETSc MCP server terminated.")
