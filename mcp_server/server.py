"""
Trusty MCP server — Chapter 21 (§9.16).
Exposes Trusty's capabilities as named MCP tools using the
Model Context Protocol Python SDK (mcp).

Runs as a separate process alongside FastAPI:
    Terminal 1: uvicorn app:app --reload
    Terminal 2: python mcp_server/server.py
    Terminal 3: python ui/app_ui.py

The server uses stdio transport by default (standard for local MCP
servers). For networked deployment, switch to SSE transport.
"""
from __future__ import annotations

import json
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from mcp_server.tools import TOOL_REGISTRY
from utils.logging import logger


def build_server() -> Server:
    server = Server("trusty")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=entry["description"],
                inputSchema={
                    "type": "object",
                    "properties": _infer_schema(entry["fn"]),
                },
            )
            for name, entry in TOOL_REGISTRY.items()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool: {name!r}")
        try:
            result = TOOL_REGISTRY[name]["fn"](**arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            logger.error(f"[mcp] Tool {name!r} failed: {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


def _infer_schema(fn) -> dict:
    """Build a minimal JSON schema from a function's type annotations."""
    import inspect
    schema = {}
    sig = inspect.signature(fn)
    for param_name, param in sig.parameters.items():
        annotation = param.annotation
        if annotation in (str, inspect.Parameter.empty):
            schema[param_name] = {"type": "string"}
        elif annotation == bool:
            schema[param_name] = {"type": "boolean"}
        elif annotation == int:
            schema[param_name] = {"type": "integer"}
        else:
            schema[param_name] = {"type": "string"}
    return schema


if __name__ == "__main__":
    import asyncio
    server = build_server()
    logger.info("[mcp] Trusty MCP server starting on stdio transport")
    asyncio.run(stdio_server(server))