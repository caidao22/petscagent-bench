import asyncio
from src.util.a2a_comm import send_message

async def main(green_url: str="http://localhost:9001", purple_url: str="http//localhost:9002"):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal the green agent to start the benchmark.")
    parser.add_argument("--green_url", type=str, default="http://localhost:9001", help="Green agent URL")
    parser.add_argument("--purple_url", type=int, default="http://localhost:9002", help="Purple agent URL")
    args = parser.parse_args()
    asyncio.run(main(args.green_url, args.purple_url))
