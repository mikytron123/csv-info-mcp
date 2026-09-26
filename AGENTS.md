# Agent Instructions

## Project Structure

- [server.py](server.py) implements the FastMCP CSV inspection tools and starts the server.
- [client.py](client.py) connects to the MCP server and Ollama for interactive queries.
- [appconfig.py](appconfig.py) defines Ollama configuration and loads `.env` values.
- [mcp.json](mcp.json) configures the client server URL and filesystem roots.

## Development

- Start the server with `python server.py --transport sse`; the client expects the configured SSE endpoint in [mcp.json](mcp.json).
- Run the client with `python client.py`. It requires a reachable MCP server and Ollama instance.
- Use `just format` for Black formatting. `just lint` runs isort, pyupgrade, autoflake, and flake8; the first three commands modify files.
- Follow the existing Python style: type annotations, async MCP tool handlers, and Black formatting with isort's Black profile.
- No test suite or dependency manifest is currently present. Do not assume a project-specific test or install command.

## Change Guidance

- Keep CSV access aligned with the roots supplied by MCP Roots or `--root_directory`; review path containment whenever changing file lookup.
- Keep Ollama settings and server connection settings in their existing configuration files rather than duplicating them.