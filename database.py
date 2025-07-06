#!/usr/bin/env python3
"""
database.py - Database class for storing call graph data in SQLite.
Uses a normalized schema: repositories have unique IDs, nodes/edges reference repo_id.
Extended with minimal summary support.
"""
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class Database:
    """Handles SQLite DB operations and schema for call graph data"""

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.create_schema()

    def create_schema(self):
        """Create tables for repositories, nodes, edges, and summaries"""
        cur = self.conn.cursor()

        # Repositories table: stores unique repo names and paths
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS repos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL
            )
        """
        )

        # Nodes: each tied to repo_id
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                key TEXT,
                repo_id INTEGER,
                name TEXT,
                actual_name TEXT,
                file TEXT,
                start INTEGER,
                end INTEGER,
                kind TEXT,
                PRIMARY KEY (key, repo_id),
                FOREIGN KEY (repo_id) REFERENCES repos(id) ON DELETE CASCADE
            )
        """
        )

        # Edges: each tied to repo_id
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id INTEGER,
                from_key TEXT,
                to_key TEXT,
                kind TEXT,
                FOREIGN KEY (repo_id) REFERENCES repos(id) ON DELETE CASCADE
            )
        """
        )

        # Summaries: LLM-generated summaries for nodes
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                node_key TEXT,
                repo_id INTEGER,
                summary_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (node_key, repo_id),
                FOREIGN KEY (node_key, repo_id) REFERENCES nodes(key, repo_id) ON DELETE CASCADE
            )
        """
        )

        self.conn.commit()

    def get_or_create_repo_id(self, repo_name: str, repo_path: str) -> int:
        """Return repo_id for the given repo, creating it if it doesn't exist"""
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
        row = cur.fetchone()
        if row:
            return row[0]

        cur.execute(
            "INSERT INTO repos (name, path) VALUES (?, ?)", (repo_name, repo_path)
        )
        self.conn.commit()
        return cur.lastrowid

    def replace_repo_data(
        self, repo_id: int, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ):
        """Replace data for a single repo: delete existing nodes/edges, insert new"""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM nodes WHERE repo_id = ?", (repo_id,))
        cur.execute("DELETE FROM edges WHERE repo_id = ?", (repo_id,))
        self.conn.commit()

        self.insert_nodes(repo_id, nodes)
        self.insert_edges(repo_id, edges)

    def insert_nodes(self, repo_id: int, nodes: List[Dict[str, Any]]):
        """Insert multiple nodes for the given repo_id"""
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO nodes (key, repo_id, name, actual_name, file, start, end, kind)
            VALUES (:key, :repo_id, :name, :actualName, :file, :start, :end, :kind)
        """,
            [{**node, "repo_id": repo_id} for node in nodes],
        )
        self.conn.commit()

    def insert_edges(self, repo_id: int, edges: List[Dict[str, Any]]):
        """Insert multiple edges for the given repo_id"""
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO edges (repo_id, from_key, to_key, kind)
            VALUES (:repo_id, :from, :to, :kind)
        """,
            [{"repo_id": repo_id, **edge} for edge in edges],
        )
        self.conn.commit()

    # Summary-related methods
    def get_repo_by_name(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """Get repo info by name"""
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, path FROM repos WHERE name = ?", (repo_name,))
        row = cur.fetchone()
        return {"id": row[0], "name": row[1], "path": row[2]} if row else None

    def get_all_nodes(self, repo_id: int) -> List[Dict[str, Any]]:
        """Get all nodes for a repository in a single efficient call"""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT key, name, actual_name, file, start, end, kind
            FROM nodes WHERE repo_id = ?
            ORDER BY key
        """,
            (repo_id,),
        )

        return [
            {
                "key": row[0],
                "name": row[1],
                "actual_name": row[2],
                "file": row[3],
                "start": row[4],
                "end": row[5],
                "kind": row[6],
            }
            for row in cur.fetchall()
        ]

    def get_all_edges(self, repo_id: int) -> List[Dict[str, Any]]:
        """Get all edges for a repository in a single efficient call"""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, from_key, to_key, kind
            FROM edges WHERE repo_id = ?
            ORDER BY id
        """,
            (repo_id,),
        )

        return [
            {"id": row[0], "from_key": row[1], "to_key": row[2], "kind": row[3]}
            for row in cur.fetchall()
        ]

    def get_all_summaries(self, repo_id: int) -> Dict[str, Dict[str, Any]]:
        """Get all summaries for a repository in a single efficient call"""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT node_key, summary_json, created_at 
            FROM summaries WHERE repo_id = ?
        """,
            (repo_id,),
        )

        summaries = {}
        for row in cur.fetchall():
            summaries[row[0]] = {"summary": json.loads(row[1]), "created_at": row[2]}
        return summaries

    def get_nodes_without_summaries(self, repo_id: int) -> List[Dict[str, Any]]:
        """Get nodes that don't have summaries yet"""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT n.key, n.name, n.actual_name, n.file, n.start, n.end, n.kind
            FROM nodes n
            LEFT JOIN summaries s ON n.key = s.node_key AND n.repo_id = s.repo_id
            WHERE n.repo_id = ? AND s.node_key IS NULL
            ORDER BY n.key ASC
        """,
            (repo_id,),
        )

        return [
            {
                "key": row[0],
                "name": row[1],
                "actual_name": row[2],
                "file": row[3],
                "start": row[4],
                "end": row[5],
                "kind": row[6],
            }
            for row in cur.fetchall()
        ]

    def get_node_dependencies(
        self, repo_id: int, node_key: str
    ) -> List[Dict[str, Any]]:
        """Get dependencies of a node (nodes it depends on)"""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT n.key, n.name, n.actual_name, n.kind
            FROM edges e
            JOIN nodes n ON e.to_key = n.key AND e.repo_id = n.repo_id
            WHERE e.from_key = ? AND e.repo_id = ? AND e.is_cycle_edge = FALSE
        """,
            (node_key, repo_id),
        )

        return [
            {"key": row[0], "name": row[1], "actual_name": row[2], "kind": row[3]}
            for row in cur.fetchall()
        ]

    def get_summary(self, repo_id: int, node_key: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific node"""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT summary_json, created_at FROM summaries 
            WHERE repo_id = ? AND node_key = ?
        """,
            (repo_id, node_key),
        )

        row = cur.fetchone()
        if row:
            return {"summary": json.loads(row[0]), "created_at": row[1]}
        return None

    def store_summary(self, repo_id: int, node_key: str, summary_dict: Dict[str, Any]):
        """Store a summary for a node"""
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO summaries (node_key, repo_id, summary_json)
            VALUES (?, ?, ?)
        """,
            (node_key, repo_id, json.dumps(summary_dict)),
        )
        self.conn.commit()

    def get_summary_stats(self, repo_id: int) -> Dict[str, int]:
        """Get summary statistics for a repository"""
        cur = self.conn.cursor()

        cur.execute("SELECT COUNT(*) FROM nodes WHERE repo_id = ?", (repo_id,))
        total_nodes = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM summaries WHERE repo_id = ?", (repo_id,))
        summarized_nodes = cur.fetchone()[0]

        return {"total_nodes": total_nodes, "summarized_nodes": summarized_nodes}
