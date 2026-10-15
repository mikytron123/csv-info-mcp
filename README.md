# csv-info-mcp

An MCP server and interactive Ollama client for inspecting CSV files. The server reads CSV files with Polars and exposes metadata tools; the client makes those tools available to an Ollama model.

## Features

The MCP server provides tools to:

- Return a CSV file's column names and inferred schema.
- Count the rows or columns in a CSV file.

CSV files are looked up by a relative path under one or more directories configured on the server with `--root-directory`. This keeps filesystem configuration on the server instead of using MCP Roots, which the [2026-07-28 MCP specification deprecates](https://modelcontextprotocol.io/specification/2026-07-28/client/roots) and recommends replacing with server configuration or paths passed as tool parameters. The server also checks resolved paths to prevent access outside the configured directories.

## Requirements

- Python with the project's runtime packages installed from [requirements.txt](requirements.txt): MCP SDK v2 (`mcp`), Polars, the Ollama Python client, `python-dotenv`, `environ-config`, and Pydantic.
- A running Ollama service and a model available locally. The default model is `qwen3:8b`.

This repository does not currently include an automated test suite.

## Configuration

Edit [mcp.json](mcp.json) to set the MCP server URL. The checked-in configuration points to `http://localhost:8000/mcp`, the Streamable HTTP endpoint. Configure directories separately when starting the server; the client no longer sends filesystem roots.

Ollama settings are read from environment variables, which can be placed in a `.env` file:

| Variable | Default |
| --- | --- |
| `OLLAMA_HOST` | `localhost` |
| `OLLAMA_MODEL` | `qwen3:8b` |

The client currently passes `OLLAMA_HOST` to the Ollama client; `OLLAMA_PORT` is declared in the configuration but is not used separately.

## Run

Start the server in one terminal, choosing the directories it may inspect:

```sh
python server.py --transport streamable-http --root-directory ./tempdir
```

Then start the interactive client in another terminal:

```sh
python client.py
```

Ask questions about CSV files in a configured server directory, for example: `How many rows are in temp.csv?` Type `quit` to exit.

The server also accepts `stdio` when started directly with `--transport`. Multiple directories can be supplied after `--root-directory`.

The client and server use MCP Python SDK v2, which implements the 2026-07-28 protocol revision. The server uses Streamable HTTP instead of the deprecated HTTP+SSE transport.

## Code Quality

Use `just format` to format Python files with Black. `just lint` runs isort, pyupgrade, autoflake, and Flake8; note that its first three steps modify files.
