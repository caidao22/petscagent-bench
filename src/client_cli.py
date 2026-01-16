import asyncio
from src.util.a2a_comm import send_message
import argparse

async def main(green_url: str="http://localhost:9001", purple_url: str="http//localhost:9002", mcp_server_url: str="http://localhost:8080/mcp", green_id="", purple_id=""):
    # send the task description
    print("Sending task description to green agent...")
    task_text = f"""
Your task is to instantiate petscagent-bench to test the agent located at:
<purple_agent_url>
{purple_url}/
</purple_agent_url>
You can use MCP tools from:
<mcp_server_url>
{mcp_server_url}/
</mcp_server_url>
Green agent's AgentBeats ID is
<green_id>
{green_id}
</green_id>
Purple agent's AgentBeats ID is
<purple_id>
{purple_id}
</purple_id>
    """
    print("Task description:")
    print(task_text)
    print("Sending...")
    response = await send_message(green_url, task_text)
    print("Response from green agent:")
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal the green agent to start the benchmark.")
    parser.add_argument("--green-url", type=str, default="http://localhost:9001", help="Green agent URL")
    parser.add_argument("--purple-url", type=str, default="http://localhost:9002", help="Purple agent URL")
    parser.add_argument("--mcp-server-url", type=str, default="http://localhost:8080/mcp", help="MCP server URL")
    parser.add_argument("--green-id", type=str, default="019bb856-c8bf-7390-8c4f-bced52276932", help="Green agent's AgentBeats ID")
    parser.add_argument("--purple-id", type=str, help="Purple agent's AgentBeats ID")
    args = parser.parse_args()
    asyncio.run(main(args.green_url, args.purple_url, args.mcp_server_url, green_id=args.green_id, purple_id=args.purple_id))
