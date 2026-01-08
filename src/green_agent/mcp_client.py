import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import os
import sys

from requests import session


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_local_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env={
                "PETSC_DIR": os.getenv("PETSC_DIR"),
                "PETSC_ARCH": os.getenv("PETSC_ARCH"),
            },
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in self.tools])

    async def connect_to_remote_server(self, server_url: str):
        read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
            streamablehttp_client(server_url)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in self.tools])

    async def run_bash_command(self, command: str, args: str = ""):
        # support "upload_file filename" and "upload_file filename data"
        if command == "upload_file":
            if os.path.isfile(args):
                filename = os.path.basename(args)
                with open(args) as fd:
                    data = fd.read()
            else:
                filename = args[0 : args.find(" ")]
                data = args[args.find(" ") :]
            args = {"filename": filename, "data": data}
        elif args:
            args = {"arg": args}
        else:
            args = {}

        if command in [tool.name for tool in self.tools]:
            response = await self.session.call_tool(command, arguments=args)
        else:
            if args:
                args["arg"] = command + " " + args["arg"]
            else:
                args = {"arg": command}
            response = await self.session.call_tool("run_bash_command", arguments=args)
        return response

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <path_to_server_script/server_url>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_local_server(sys.argv[1])
        # await client.connect_to_remote_server(sys.argv[1])
        # testing tool calling
        # await client.run_bash_command("upload_file", "generated_codes/Robertson_ODE.c")
        # await client.run_bash_command("make", "Robertson_ODE")
        # result = await client.run_bash_command("./Robertson_ODE","-ts_type rosw -ts_monitor -ts_adapt_type basic")
        result = await client.run_bash_command("ls -l")
        print(result.content[0].text)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
