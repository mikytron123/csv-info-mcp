import sys
from typing import Literal, Optional
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import ClientCapabilities, RootsCapability
import argparse
from pathlib import Path
import polars as pl

# Create an MCP server
mcp = FastMCP("CSVInfo")

allowed_directories: Optional[list[str]] = None


async def get_directories(ctx: Context) -> list[str]:
    """Get directories for searching CSVs,
    use list root callback if the client is capable,
    Otherwise fall back to cli arguments
    """
    global allowed_directories

    if allowed_directories is not None:
        return allowed_directories

    cap = ctx.session.check_client_capability(
        ClientCapabilities(roots=RootsCapability())
    )
    if cap:
        list_roots = await ctx.session.list_roots()
        dirs = [root.uri.path for root in list_roots.roots if root.uri.path is not None]
        if len(dirs) == 0:
            raise ValueError("No root directories available from client.")
        allowed_directories = dirs

        return allowed_directories

    args = parse_args()
    if args.root_directory is not None:
        allowed_directories = [args.root_directory[0]]
        return allowed_directories

    raise ValueError(
        "No root directories available. Please provide a root directory using --root_directory argument or ensure the client supports RootsCapability."
    )


def find_file_in_allowed_dirs(file_path: str, allowed_dirs: list[str]) -> Optional[str]:
    """Search for the file in the allowed directories and return its full path if found."""
    for dir in allowed_dirs:
        potential_path = Path(dir) / file_path
        if potential_path.exists() and potential_path.is_file():
            return str(potential_path)
    return None


def read_csv(file_path: str) -> pl.DataFrame:
    """Reads a csv into a polars dataframe"""
    df = pl.read_csv(str(file_path))
    return df


@mcp.tool()
async def get_csv_schema(file_path: str, ctx: Context) -> dict:
    """Get the schema of a CSV file"""
    try:
        dirs = await get_directories(ctx)
        full_file_path = find_file_in_allowed_dirs(file_path, dirs)
        if full_file_path is None:
            raise ValueError(f"File not found in allowed directories: {dirs}")
        df = read_csv(full_file_path)
        schema = {col: str(dtype) for col, dtype in df.schema.items()}
        return schema
    except Exception as e:
        raise ValueError(f"Error getting CSV schema: {e}")


@mcp.tool()
async def count_csv_columns(file_path: str, ctx: Context) -> int:
    """Count the number of columns in a CSV file"""
    try:
        dirs = await get_directories(ctx)
        full_file_path = find_file_in_allowed_dirs(file_path, dirs)
        if full_file_path is None:
            raise ValueError(f"File not found in allowed directories: {dirs}")
        df = read_csv(full_file_path)
        return len(df.columns)
    except Exception as e:
        raise ValueError(f"Error counting CSV rows: {e}")


@mcp.tool()
async def count_csv_rows(file_path: str, ctx: Context) -> int:
    """Count the number of rows in a CSV file"""
    try:
        dirs = await get_directories(ctx)
        full_file_path = find_file_in_allowed_dirs(file_path, dirs)
        if full_file_path is None:
            raise ValueError(f"File not found in allowed directories: {dirs}")
        df = read_csv(full_file_path)
        return len(df)
    except Exception as e:
        raise ValueError(f"Error counting CSV rows: {e}")


@mcp.tool()
async def read_csv_columns(file_path: str, ctx: Context) -> list[str]:
    """Read a CSV file and return its column names"""
    try:
        dirs = await get_directories(ctx)
        full_file_path = find_file_in_allowed_dirs(file_path, dirs)
        if full_file_path is None:
            raise ValueError(f"File not found in allowed directories: {dirs}")
        df = read_csv(full_file_path)
        return df.columns
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def validate_directory_path(path: str) -> Path:
    """
    Validate and normalize directory path using pathlib

    Args:
        path: The directory path to validate

    Returns:
        Path object representing the validated absolute directory path

    Raises:
        ValueError: If the path doesn't exist or is not a directory
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise ValueError(f"Directory does not exist: {path}")

    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    return path_obj.resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="FastMCP Server")
    parser.add_argument(
        "--root_directory",
        nargs=1,
        required=False,
        type=str,
        help="Root directory for file operations",
    )
    parser.add_argument("-t", "--transport", type=str, help="Transport method")
    return parser.parse_args()


def main(transport: Literal["stdio", "sse", "streamable-http"]):
    """Entry point for the direct execution server."""
    mcp.run(transport=transport)


if __name__ == "__main__":
    args = parse_args()

    if args.transport not in ["stdio", "sse", "streamable-http"]:
        print(
            f"Error: Unsupported transport '{args.transport}'. Supported transports are 'stdio', 'sse', and 'streamable-http'."
        )
        sys.exit(1)

    main(args.transport)
