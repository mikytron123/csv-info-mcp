# csv-info-mcp

An MCP server and interactive Ollama client for inspecting CSV files. The server reads CSV files with Polars and exposes metadata tools; the client makes those tools available to an Ollama model.

## Features

The MCP server provides tools to:

- Return a CSV file's column names and inferred schema.
- Count the rows or columns in a CSV file.

CSV files are looked up by a path relative to a directory made available through the client's MCP Roots capability. The server can also use `--root_directory` when the client does not provide roots.

## Requirements

- Python with the project's runtime packages installed: MCP (`mcp`), Polars, the Ollama Python client, `python-dotenv`, `environ-config`, and Pydantic.
- A running Ollama service and a model available locally. The default model is `qwen3:8b`.

This repository does not currently include a dependency manifest or automated test suite.

## Configuration

Edit [mcp.json](mcp.json) to set the MCP server URL and the directories the client will expose to the server. The checked-in configuration points to `http://localhost:8000/sse` and the local `tempdir/`; update its file URI if your workspace is elsewhere.

Ollama settings are read from environment variables, which can be placed in a `.env` file:

| Variable | Default |
| --- | --- |
| `OLLAMA_HOST` | `localhost` |
| `OLLAMA_MODEL` | `qwen3:8b` |

The client currently passes `OLLAMA_HOST` to the Ollama client; `OLLAMA_PORT` is declared in the configuration but is not used separately.

## Run

Start the server in one terminal:

```sh
python server.py --transport sse
```

Then start the interactive client in another terminal:

```sh
python client.py
```

Ask questions about CSV files in a configured root, for example: `How many rows are in temp.csv?` Type `quit` to exit.

The server also accepts `stdio` and `streamable-http` transports when started directly with `--transport`.

## Code Quality

Use `just format` to format Python files with Black. `just lint` runs isort, pyupgrade, autoflake, and Flake8; note that its first three steps modify files.
