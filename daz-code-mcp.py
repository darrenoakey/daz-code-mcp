#!/usr/bin/env python3
"""
MCP Server for Code Repository ChromaDB Search
Provides search_code_snippets and search_code_full functions via Model Context Protocol
"""

import json
import logging
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, ServerCapabilities, ToolsCapability
from mcp.server.stdio import stdio_server

# Set up logging to stderr so it doesn't interfere with stdio communication
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RESULTS_LIMIT = 10


class CodeSearchServer:
    def __init__(self, db_path: Path):
        self.db_path = db_path

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False)
        )

        # Get all collections (one per repository)
        self.collections = {}
        try:
            collection_list = self.client.list_collections()
            for collection in collection_list:
                if collection.name.startswith("code_chunks_"):
                    repo_name = collection.name.replace("code_chunks_", "")
                    self.collections[repo_name] = collection
            logger.info(f"Connected to {len(self.collections)} code repositories: {list(self.collections.keys())}")
        except Exception as e:
            logger.error(f"Failed to connect to collections: {e}")
            raise

    def search_code_snippets(
        self, query: str, limit: int = DEFAULT_RESULTS_LIMIT
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant code chunks and return them
        """
        try:
            all_results = []
            
            # Search across all collections
            for repo, collection in self.collections.items():
                # Perform vector search
                results = collection.query(
                    query_texts=[query],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                )

                if results["ids"][0]:
                    # Format results
                    for i in range(len(results["ids"][0])):
                        snippet = {
                            "chunk_id": results["ids"][0][i],
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                            "relevance_score": 1 - results["distances"][0][i],
                            "repo_name": repo
                        }
                        all_results.append(snippet)

            # Sort all results by relevance and limit
            all_results.sort(key=lambda x: x["distance"])
            return all_results[:limit]

        except Exception as e:
            logger.error(f"Error searching code snippets: {e}")
            return []

    def search_functions(
        self, query: str, limit: int = DEFAULT_RESULTS_LIMIT
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for functions
        """
        try:
            all_results = []
            
            # Search across all collections for functions only
            for repo, collection in self.collections.items():
                results = collection.query(
                    query_texts=[query],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                    where={"chunk_type": "function"}
                )

                if results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        snippet = {
                            "chunk_id": results["ids"][0][i],
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                            "relevance_score": 1 - results["distances"][0][i],
                            "repo_name": repo
                        }
                        all_results.append(snippet)

            # Sort all results by relevance and limit
            all_results.sort(key=lambda x: x["distance"])
            return all_results[:limit]

        except Exception as e:
            logger.error(f"Error searching functions: {e}")
            return []

    def search_classes(
        self, query: str, limit: int = DEFAULT_RESULTS_LIMIT
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for classes
        """
        try:
            all_results = []
            
            # Search across all collections for classes only
            for repo, collection in self.collections.items():
                results = collection.query(
                    query_texts=[query],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                    where={"chunk_type": "class"}
                )

                if results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        snippet = {
                            "chunk_id": results["ids"][0][i],
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                            "relevance_score": 1 - results["distances"][0][i],
                            "repo_name": repo
                        }
                        all_results.append(snippet)

            # Sort all results by relevance and limit
            all_results.sort(key=lambda x: x["distance"])
            return all_results[:limit]

        except Exception as e:
            logger.error(f"Error searching classes: {e}")
            return []

    def search_files(
        self, query: str, limit: int = DEFAULT_RESULTS_LIMIT
    ) -> List[Dict[str, Any]]:
        """
        Search for files by filename
        """
        try:
            all_results = []
            
            # Search across all collections for filenames only
            for repo, collection in self.collections.items():
                results = collection.query(
                    query_texts=[query],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                    where={"chunk_type": "filename"}
                )

                if results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        snippet = {
                            "chunk_id": results["ids"][0][i],
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                            "relevance_score": 1 - results["distances"][0][i],
                            "repo_name": repo
                        }
                        all_results.append(snippet)

            # Sort all results by relevance and limit
            all_results.sort(key=lambda x: x["distance"])
            return all_results[:limit]

        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []

    def search_code_full(
        self, query: str, limit: int = DEFAULT_RESULTS_LIMIT
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant code chunks and return full file contents
        """
        try:
            all_results = []
            
            # Search across all collections
            for repo, collection in self.collections.items():
                # First get relevant chunks (excluding filename chunks for this search)
                chunk_results = collection.query(
                    query_texts=[query],
                    n_results=limit * 3,  # Get more chunks to find unique files
                    include=["metadatas", "distances"],
                    where={"chunk_type": {"$ne": "filename"}}
                )

                if not chunk_results["ids"][0]:
                    continue

                # Collect unique file paths with their best distances
                file_scores = {}
                for i in range(len(chunk_results["ids"][0])):
                    metadata = chunk_results["metadatas"][0][i]
                    file_path = metadata["file_path"]
                    distance = chunk_results["distances"][0][i]

                    if file_path not in file_scores or distance < file_scores[file_path]:
                        file_scores[file_path] = distance

                # Sort by distance and add to results
                sorted_files = sorted(file_scores.items(), key=lambda x: x[1])
                
                for file_path, best_distance in sorted_files:
                    # Get the filename chunk for this file
                    file_chunks = collection.get(
                        where={
                            "file_path": file_path,
                            "chunk_type": "filename"
                        },
                        include=["documents", "metadatas"]
                    )

                    if file_chunks["ids"]:
                        # Use the filename chunk which contains the full file content
                        article = {
                            "file_path": file_path,
                            "content": file_chunks["documents"][0],
                            "metadata": file_chunks["metadatas"][0],
                            "relevance_score": 1 - best_distance,
                            "repo_name": repo
                        }
                        all_results.append(article)

            # Sort all results by relevance and limit
            all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return all_results[:limit]

        except Exception as e:
            logger.error(f"Error searching full code files: {e}")
            return []


# Initialize global variables
app = Server("code-search")
search_server: Optional[CodeSearchServer] = None


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="search_code_snippets",
            description="Search for relevant code chunks/snippets from repositories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": DEFAULT_RESULTS_LIMIT,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_functions",
            description="Search specifically for functions in code repositories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": DEFAULT_RESULTS_LIMIT,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_classes",
            description="Search specifically for classes in code repositories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": DEFAULT_RESULTS_LIMIT,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_files",
            description="Search for files by filename in code repositories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string (filename)"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": DEFAULT_RESULTS_LIMIT,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_code_full",
            description="Search for relevant code and return full file content",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of files to return",
                        "default": DEFAULT_RESULTS_LIMIT,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    global search_server

    if search_server is None:
        # Initialize search server on first use
        script_dir = Path(__file__).parent
        db_path = script_dir / "code_chromadb"

        if not db_path.exists():
            return [
                TextContent(
                    type="text",
                    text="Error: Code ChromaDB database not found. Please run the code scanner first.",
                )
            ]

        try:
            search_server = CodeSearchServer(db_path)
        except Exception as e:
            return [
                TextContent(
                    type="text", text=f"Error initializing code search server: {str(e)}"
                )
            ]

    if name == "search_code_snippets":
        query = arguments.get("query", "")
        limit = arguments.get("limit", DEFAULT_RESULTS_LIMIT)

        results = search_server.search_code_snippets(query, limit)

        if not results:
            return [TextContent(type="text", text="No results found for your query.")]

        # Format results for display
        formatted_results = []
        for i, result in enumerate(results):
            metadata = result['metadata']
            formatted = f"**Result {i+1}:**\n"
            formatted += f"Repository: {result['repo_name']}\n"
            formatted += f"File: {metadata['file_path']}\n"
            formatted += f"Type: {metadata.get('chunk_type', 'unknown')}\n"
            
            if metadata.get('element_name'):
                formatted += f"Element: {metadata['element_name']}\n"
            if metadata.get('language'):
                formatted += f"Language: {metadata['language']}\n"
            
            formatted += f"Relevance: {result['relevance_score']:.3f}\n"
            formatted += f"Content:\n{result['content']}\n"
            formatted += "-" * 60
            formatted_results.append(formatted)

        return [TextContent(type="text", text="\n".join(formatted_results))]

    elif name == "search_functions":
        query = arguments.get("query", "")
        limit = arguments.get("limit", DEFAULT_RESULTS_LIMIT)

        results = search_server.search_functions(query, limit)

        if not results:
            return [TextContent(type="text", text="No function results found for your query.")]

        # Format results for display
        formatted_results = []
        for i, result in enumerate(results):
            metadata = result['metadata']
            formatted = f"**Function {i+1}: {metadata.get('element_name', 'Unknown')}**\n"
            formatted += f"Repository: {result['repo_name']}\n"
            formatted += f"File: {metadata['file_path']}\n"
            if metadata.get('language'):
                formatted += f"Language: {metadata['language']}\n"
            formatted += f"Relevance: {result['relevance_score']:.3f}\n"
            formatted += f"\n{result['content']}\n"
            formatted += "-" * 60
            formatted_results.append(formatted)

        return [TextContent(type="text", text="\n".join(formatted_results))]

    elif name == "search_classes":
        query = arguments.get("query", "")
        limit = arguments.get("limit", DEFAULT_RESULTS_LIMIT)

        results = search_server.search_classes(query, limit)

        if not results:
            return [TextContent(type="text", text="No class results found for your query.")]

        # Format results for display
        formatted_results = []
        for i, result in enumerate(results):
            metadata = result['metadata']
            formatted = f"**Class {i+1}: {metadata.get('element_name', 'Unknown')}**\n"
            formatted += f"Repository: {result['repo_name']}\n"
            formatted += f"File: {metadata['file_path']}\n"
            if metadata.get('language'):
                formatted += f"Language: {metadata['language']}\n"
            formatted += f"Relevance: {result['relevance_score']:.3f}\n"
            formatted += f"\n{result['content']}\n"
            formatted += "-" * 60
            formatted_results.append(formatted)

        return [TextContent(type="text", text="\n".join(formatted_results))]

    elif name == "search_files":
        query = arguments.get("query", "")
        limit = arguments.get("limit", DEFAULT_RESULTS_LIMIT)

        results = search_server.search_files(query, limit)

        if not results:
            return [TextContent(type="text", text="No file results found for your query.")]

        # Format results for display
        formatted_results = []
        for i, result in enumerate(results):
            metadata = result['metadata']
            formatted = f"**File {i+1}: {metadata.get('element_name', 'Unknown')}**\n"
            formatted += f"Repository: {result['repo_name']}\n"
            formatted += f"Path: {metadata['file_path']}\n"
            formatted += f"Relevance: {result['relevance_score']:.3f}\n"
            formatted += f"\nContent preview:\n{result['content'][:500]}...\n"
            formatted += "-" * 60
            formatted_results.append(formatted)

        return [TextContent(type="text", text="\n".join(formatted_results))]

    elif name == "search_code_full":
        query = arguments.get("query", "")
        limit = arguments.get("limit", DEFAULT_RESULTS_LIMIT)

        results = search_server.search_code_full(query, limit)

        if not results:
            return [TextContent(type="text", text="No full file results found for your query.")]

        # Format results for display
        formatted_results = []
        for i, result in enumerate(results):
            metadata = result['metadata']
            formatted = f"**File {i+1}: {metadata.get('element_name', 'Unknown')}**\n"
            formatted += f"Repository: {result['repo_name']}\n"
            formatted += f"Path: {result['file_path']}\n"
            formatted += f"Relevance Score: {result['relevance_score']:.3f}\n"
            formatted += f"\n{result['content']}\n"
            formatted += "=" * 80
            formatted_results.append(formatted)

        return [TextContent(type="text", text="\n\n".join(formatted_results))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server"""
    logger.info("Starting Code Search MCP Server...")

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="code-search",
                server_version="1.0.0",
                capabilities=ServerCapabilities(tools=ToolsCapability()),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())