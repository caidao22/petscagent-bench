"""Launcher module - initiates and coordinates the evaluation process."""

import multiprocessing
import json
from src.green_agent.agent import start_green_agent
from src.white_agent.petsc_agent import start_white_agent
from src.util.my_a2a import wait_agent_ready, send_message


async def launch_evaluation():
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

    # start white agent
    print("Launching white agent...")
    white_address = ("localhost", 9002)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
    p_white = multiprocessing.Process(
        target=start_white_agent, args=("general_white_agent", *white_address)
    )
    p_white.start()
    assert await wait_agent_ready(white_url), "White agent not ready in time"
    print("White agent is ready.")

    # send the task description
    print("Sending task description to green agent...")
    task_text = f"""
Your task is to instantiate petscagent-bench to test the agent located at:
<white_agent_url>
http://{white_address[0]}:{white_address[1]}/
</white_agent_url>
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
    p_white.terminate()
    p_white.join()
    print("Agents terminated.")
