import argparse
from pathlib import Path
from typing import Literal

import polars as pl
from mcp.server import MCPServer

mcp = MCPServer("CSVInfo")
allowed_directories: list[Path] = []


def find_file_in_allowed_dirs(file_path: str, allowed_dirs: list[Path]) -> Path | None:
    """Find a relative file path within the configured directories."""
    requested_path = Path(file_path)
    if requested_path.is_absolute():
        raise ValueError("file_path must be relative to a configured directory")

    for directory in allowed_dirs:
        try:
            potential_path = (directory / requested_path).resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError):
            continue

        if potential_path.is_relative_to(directory) and potential_path.is_file():
            return potential_path

    return None


def get_csv_file(file_path: str) -> Path:
    """Resolve a CSV path within the server's configured directories."""
    if not allowed_directories:
        raise RuntimeError(
            "No CSV directories configured; start the server with --root-directory."
        )

    full_file_path = find_file_in_allowed_dirs(file_path, allowed_directories)
    if full_file_path is None:
        raise ValueError(f"CSV file not found in configured directories: {file_path}")
    return full_file_path


def read_csv(file_path: Path) -> pl.DataFrame:
    """Read a CSV into a Polars dataframe."""
    return pl.read_csv(file_path)


@mcp.tool()
async def get_csv_schema(file_path: str) -> dict[str, str]:
    """Get the schema of a CSV file relative to a configured server directory."""
    df = read_csv(get_csv_file(file_path))
    return {col: str(dtype) for col, dtype in df.schema.items()}


@mcp.tool()
async def count_csv_columns(file_path: str) -> int:
    """Count the columns in a CSV file relative to a configured server directory."""
    df = read_csv(get_csv_file(file_path))
    return len(df.columns)


@mcp.tool()
async def count_csv_rows(file_path: str) -> int:
    """Count the rows in a CSV file relative to a configured server directory."""
    df = read_csv(get_csv_file(file_path))
    return len(df)


@mcp.tool()
async def read_csv_columns(file_path: str) -> list[str]:
    """Read column names from a CSV file relative to a configured server directory."""
    df = read_csv(get_csv_file(file_path))
    return df.columns


def validate_directory_path(path: str) -> Path:
    """Return the resolved path if it names an existing directory."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"Directory does not exist: {path}")
    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    return path_obj.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSVInfo MCP server")
    parser.add_argument(
        "--root-directory",
        "--root_directory",
        dest="root_directories",
        nargs="+",
        required=True,
        type=str,
        help="Directory or directories the server may use for CSV files",
    )
    parser.add_argument(
        "-t",
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport method",
    )
    return parser.parse_args()


def main(
    transport: Literal["stdio", "streamable-http"], root_directories: list[str]
) -> None:
    """Start the server with its filesystem access configured locally."""
    global allowed_directories
    allowed_directories = [
        validate_directory_path(directory) for directory in root_directories
    ]
    mcp.run(transport=transport)


if __name__ == "__main__":
    args = parse_args()
    main(args.transport, args.root_directories)
