#!/usr/bin/env python3
"""
Call Graph Runner & Indexer
Runs call_graph on a code repository, stores nodes and edges into SQLite,
and updates relationships into the existing ChromaDB index.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings
from database import Database

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CALL_GRAPH_TOOL = Path(__file__).parent / "callgraph_runner" / "callgraph"
CHROMA_DB_DIR = Path(__file__).parent / "code_chromadb"
SQLITE_DB_FILE = Path(__file__).parent / "code_callgraph.sqlite"
CHUNK_COLLECTION_PREFIX = "code_chunks"


def run_call_graph(repo_path: Path) -> dict:
    """Run call_graph on the repo and parse JSON output"""
    try:
        result = subprocess.run(
            [str(CALL_GRAPH_TOOL), str(repo_path), "-f", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"call_graph failed: {e.stderr}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse call_graph output: {e}")
        sys.exit(1)


def update_chromadb_edges(
    client: chromadb.ClientAPI, collection_name: str, edges: list
):
    """Insert call graph edges into ChromaDB, using `from_key` and `to_key` as documents"""
    try:
        collection = client.get_collection(collection_name)
    except:
        logger.error(
            f"ChromaDB collection '{collection_name}' not found. Did you run initial indexing?"
        )
        sys.exit(1)

    # Deduplicate edges and count occurrences
    edge_counts = {}
    for edge in edges:
        edge_key = (edge["from"], edge["to"], edge["kind"])
        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

    ids, documents, metadatas = [], [], []

    for (from_key, to_key, kind), count in edge_counts.items():
        edge_id = f"{from_key}->{to_key}:{kind}"
        ids.append(edge_id)
        documents.append(
            f"CallGraphEdge: {from_key} -> {to_key} ({kind})"
            + (f" [occurs {count} times]" if count > 1 else "")
        )
        metadatas.append(
            {
                "from_key": from_key,
                "to_key": to_key,
                "kind": kind,
                "chunk_type": "callgraph_edge",
                "occurrence_count": count,
            }
        )

    # Clear existing edges for this repo first (optional - depends on your needs)
    # You might want to add a repo_id to metadata and filter by that

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        try:
            collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        except Exception as e:
            # If there are still duplicates, use upsert instead
            logger.warning(f"Add failed, trying upsert: {e}")
            collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    logger.info(
        f"Inserted {len(ids)} unique call graph edges into ChromaDB (from {len(edges)} total edges)."
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python callgraph_indexer.py <repo_path> [repo_name]")
        sys.exit(1)

    repo_path = Path(sys.argv[1]).resolve()
    repo_name = sys.argv[2] if len(sys.argv) > 2 else repo_path.name
    collection_name = f"{CHUNK_COLLECTION_PREFIX}_{repo_name}"

    logger.info(f"Running call_graph for repository at {repo_path}...")
    graph_data = run_call_graph(repo_path)

    logger.info("Updating SQLite database with call graph data...")
    db = Database(SQLITE_DB_FILE)
    repo_id = db.get_or_create_repo_id(repo_name, str(repo_path))
    db.replace_repo_data(
        repo_id, graph_data.get("nodes", []), graph_data.get("edges", [])
    )
    logger.info(
        f"Stored {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('edges', []))} edges for repo '{repo_name}' (ID: {repo_id}) in SQLite."
    )

    logger.info(
        f"Updating ChromaDB collection '{collection_name}' with call graph edges..."
    )
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR), settings=Settings(anonymized_telemetry=False)
    )
    update_chromadb_edges(client, collection_name, graph_data.get("edges", []))

    logger.info("Call graph indexing complete.")


if __name__ == "__main__":
    main()
