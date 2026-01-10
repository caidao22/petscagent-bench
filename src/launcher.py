import multiprocessing
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

async def launch_evaluation():
    """Launcher module - initiates and coordinates the evaluation process."""
    # start green agent
    print("Launching green agent...")
    green_address = ("localhost", 9001)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("green_agent", *green_address)
    )
    p_green.start()
    assert await wait_agent_ready(green_url), "Green agent not ready in time"
    print("Green agent is ready.")

    # start purple agent
    print("Launching purple agent...")
    purple_address = ("localhost", 9002)
    purple_url = f"http://{purple_address[0]}:{purple_address[1]}"
    p_purple = multiprocessing.Process(
        target=start_purple_agent, args=("general_purple_agent", *purple_address)
    )
    p_purple.start()
    assert await wait_agent_ready(purple_url), "purple agent not ready in time"
    print("purple agent is ready.")

     # start the MCP server for green agent
    print("Launching MCP server for green agent...")
    mcp_server_process = multiprocessing.Process(
        target=start_mcp_server, args=()
    )
    mcp_server_process.start()
    print("MCP server is ready.")

    # send the task description
    print("Sending task description to green agent...")
    task_text = f"""
Your task is to instantiate petscagent-bench to test the agent located at:
<purple_agent_url>
http://{purple_address[0]}:{purple_address[1]}/
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
