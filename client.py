import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult, TextContent, Tool
from ollama import AsyncClient

from appconfig import config

load_dotenv()

OLLAMA_HOST = config.ollama_host
OLLAMA_PORT = config.ollama_port
OLLAMA_MODEL = config.ollama_model


def convert_tools_to_ollama_format(tool: Tool) -> dict[str, Any]:
    """Convert an MCP tool to Ollama's tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


class MCPClient:
    def __init__(self, filepath: str = "mcp.json"):
        self.client: Client | None = None
        self.exit_stack = AsyncExitStack()
        self.ollama = AsyncClient(host=OLLAMA_HOST)
        self.config = self.read_config(filepath)

    def read_config(self, filepath: str) -> dict[str, Any]:
        """Read configuration from mcp.json."""
        with open(filepath) as config_file:
            data = json.load(config_file)

        servers = data["mcpServers"]
        return servers[next(iter(servers))]

    def get_config_url(self) -> str:
        """Return the configured Streamable HTTP URL."""
        url = self.config["url"]
        if not isinstance(url, str):
            raise ValueError("The configured MCP server URL must be a string")
        return url

    async def connect_to_server(self, server: str | StdioServerParameters) -> None:
        """Connect to an MCP server over Streamable HTTP or stdio."""
        if isinstance(server, str) and server.endswith((".py", ".js")):
            is_python = server.endswith(".py")
            server = StdioServerParameters(
                command=sys.executable if is_python else "node",
                args=[server],
                env=None,
            )

        self.client = await self.exit_stack.enter_async_context(Client(server))
        response = await self.client.list_tools()
        print(
            "\nConnected to server with tools:", [tool.name for tool in response.tools]
        )

    @staticmethod
    def get_tool_result_text(result: CallToolResult) -> str:
        """Convert an MCP tool result to text for Ollama."""
        if result.structured_content is not None:
            return json.dumps(
                result.structured_content, ensure_ascii=False, default=str
            )

        text_content = [
            item.text for item in result.content if isinstance(item, TextContent)
        ]
        if text_content:
            return "\n".join(text_content)
        if result.content:
            raise ValueError("Unsupported content type from tool")
        return "Tool returned no content"

    async def process_query(self, query: str) -> str:
        """Process a query using Ollama and available MCP tools."""
        if self.client is None:
            raise RuntimeError("Not connected to any MCP server")

        messages: list[dict[str, object]] = [{"role": "user", "content": query}]
        response = await self.client.list_tools()
        ollama_tools = [convert_tools_to_ollama_format(tool) for tool in response.tools]

        response = await self.ollama.chat(
            model=OLLAMA_MODEL, messages=messages, tools=ollama_tools, think=False
        )

        final_text = []
        message = response.message
        if message.content:
            final_text.append(message.content)

        if message.tool_calls:
            messages.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                result = await self.client.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                messages.append(
                    {
                        "role": "tool",
                        "content": self.get_tool_result_text(result),
                        "tool_name": tool_name,
                    }
                )

            response = await self.ollama.chat(
                model=OLLAMA_MODEL, messages=messages, think=False
            )
            final_text.append(response.message.content)

        return "\n".join(final_text)

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
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
                print(f"\nError: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.exit_stack.aclose()


async def main() -> None:
    client = MCPClient()
    try:
        await client.connect_to_server(client.get_config_url())
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
