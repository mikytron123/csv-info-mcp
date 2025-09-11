import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from pydantic import FileUrl
import mcp
from mcp.types import Root, ListRootsResult
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from ollama import AsyncClient
from dotenv import load_dotenv
from appconfig import config
import traceback

load_dotenv()  # load environment variables from .env

OLLAMA_HOST = config.ollama_host
OLLAMA_PORT = config.ollama_port
OLLAMA_MODEL = config.ollama_model

def convert_tools_to_ollama_format(tool: mcp.Tool):
    """Convert MCP tool to Ollama tool format"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "required": tool.inputSchema["required"],
                "properties": {
                    k: {"type": v["type"]}
                    for k, v in tool.inputSchema["properties"].items()
                },
            },
        },
    }


async def list_root_callback(context) -> ListRootsResult:
    return ListRootsResult(
        roots=[
            Root(
                uri=FileUrl("file:///home/miky/mcp-project/tempdir"),
                name="Test Root 1",
            )
        ]
    )


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.ollama = AsyncClient(host=OLLAMA_HOST, port=OLLAMA_PORT)

    async def connect_to_sse_server(self, url: str):
        """Connect to an MCP server over SSE

        Args:
            url: URL of the SSE server
        """
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(url, timeout=60)
        )
        self.stdio, self.write = sse_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                self.stdio, self.write, list_roots_callback=list_root_callback
            )
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def connect_to_server(self, server_script_path: str):
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
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.sse, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.sse, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        assert self.session is not None, "Not connected to any MCP server"
        messages = [{"role": "user", "content": query}]

        response = await self.session.list_tools()

        ollama_tools = [convert_tools_to_ollama_format(tool) for tool in response.tools]

        response = await self.ollama.chat(
            model=OLLAMA_MODEL, messages=messages, tools=ollama_tools, think=False
        )

        # Process response and handle tool calls
        final_text = []
        print(response)

        message = response.message
        if message.content:
            final_text.append(message.content)

        elif message.tool_calls:
            tool_name = ""
            for tool in message.tool_calls:
                tool_name = tool.function.name
                tool_args = tool.function.arguments

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)  # type: ignore
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

            # Continue conversation with tool results
            if message.content:
                messages.append({"role": "assistant", "content": message.content})
            print(result)

            if result.structuredContent is not None:
                messages.append(
                    {
                        "role": "tool",
                        "content": str(result.structuredContent["result"]),
                        "tool_name": tool_name,
                    }
                )

            elif isinstance(result.content[0], mcp.types.TextContent):
                messages.append(
                    {
                        "role": "tool",
                        "content": result.content[0].text,
                        "tool_name": tool_name,
                    }
                )
            else:
                raise ValueError("Unsupported content type from tool")

            print("Messages:", messages)
            response = await self.ollama.chat(
                model=OLLAMA_MODEL, messages=messages, think=False
            )

            final_text.append(response.message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(traceback.format_exc())
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python client.py <path_to_server_script>")
    #     sys.exit(1)

    client = MCPClient()
    try:
        # await client.connect_to_server(sys.argv[1])
        await client.connect_to_sse_server("http://localhost:8000/sse")
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":

    asyncio.run(main())
