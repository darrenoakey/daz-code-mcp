#!/usr/bin/env python3
"""
create_summaries.py - Generate LLM summaries for code elements in call graph database

This program:
1. Reads the call graph from the database
2. Computes topology levels (handling cycles)
3. Generates LLM summaries for nodes in dependency order
4. Stores summaries back to the database
5. Runs until killed or all nodes are summarized

Database: Uses hardwired database file 'code_callgraph.sqlite'
Saves summaries one at a time, so it can be safely interrupted and restarted.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict, deque
from pydantic import BaseModel

from colorama import init, Fore, Back, Style
from dazllm import Llm, ModelType
from database import Database
import chromadb
from chromadb.config import Settings

# Initialize colorama
init(autoreset=True)


class CodeSummary(BaseModel):
    """Structured output schema for LLM code summaries"""

    purpose: str  # What this code element does
    dependencies: List[str]  # Key dependencies it relies on
    complexity: str  # Simple/Medium/Complex
    notes: Optional[str] = None  # Additional insights


class CallGraphProcessor:
    """Processes the complete call graph in memory for efficiency"""

    def __init__(self, db: Database, repo_id: int, repo_name: str, repo_path: str):
        self.db = db
        self.repo_id = repo_id
        self.repo_name = repo_name
        self.repo_path = repo_path

        # Load everything into memory for efficiency
        print(f"{Fore.CYAN}📊 Loading call graph from database...")

        self.nodes = {node["key"]: node for node in db.get_all_nodes(repo_id)}
        self.edges = db.get_all_edges(repo_id)
        self.summaries = db.get_all_summaries(repo_id)

        print(
            f"{Fore.GREEN}✅ Loaded {len(self.nodes)} nodes, {len(self.edges)} edges, {len(self.summaries)} existing summaries"
        )

        # Build adjacency structures
        self.dependencies = defaultdict(list)  # node -> list of nodes it depends on
        self.dependents = defaultdict(list)  # node -> list of nodes that depend on it
        self.edge_lookup = {}  # (from, to) -> edge_id

        print(f"{Fore.CYAN}🔗 Building dependency graph from edges...")

        # Debug: Check first few edges and nodes
        print(
            f"{Fore.CYAN}📊 Sample node keys (first 5): {list(self.nodes.keys())[:5]}"
        )
        print(
            f"{Fore.CYAN}📊 Sample edge from_keys (first 5): {[edge['from_key'] for edge in self.edges[:5]]}"
        )
        print(
            f"{Fore.CYAN}📊 Sample edge to_keys (first 5): {[edge['to_key'] for edge in self.edges[:5]]}"
        )

        edges_processed = 0
        edges_matched = 0
        edges_from_missing = 0
        edges_to_missing = 0

        for edge in self.edges:
            edges_processed += 1
            from_key, to_key = edge["from_key"], edge["to_key"]

            from_exists = from_key in self.nodes
            to_exists = to_key in self.nodes

            if not from_exists:
                edges_from_missing += 1
                if edges_from_missing <= 3:  # Show first few missing
                    print(
                        f"{Fore.YELLOW}⚠️  Edge from_key not found in nodes: '{from_key}'"
                    )

            if not to_exists:
                edges_to_missing += 1
                if edges_to_missing <= 3:  # Show first few missing
                    print(f"{Fore.YELLOW}⚠️  Edge to_key not found in nodes: '{to_key}'")

            if from_exists and to_exists:
                self.dependencies[from_key].append(to_key)
                self.dependents[to_key].append(from_key)
                self.edge_lookup[(from_key, to_key)] = edge["id"]
                edges_matched += 1

            # Show progress
            if edges_processed % 10000 == 0:
                print(
                    f"{Fore.CYAN}   📈 Processed {edges_processed}/{len(self.edges)} edges, matched: {edges_matched}"
                )

        print(f"{Fore.CYAN}📊 Edge processing complete:")
        print(f"{Fore.CYAN}   Total edges: {edges_processed}")
        print(f"{Fore.CYAN}   Matched edges: {edges_matched}")
        print(f"{Fore.CYAN}   Missing from_key: {edges_from_missing}")
        print(f"{Fore.CYAN}   Missing to_key: {edges_to_missing}")

        if edges_matched == 0 and edges_processed > 0:
            print(
                f"{Fore.RED}💀 CRITICAL: No edges matched! This suggests node/edge key mismatch."
            )
            print(f"{Fore.RED}   This will result in all nodes being level 0.")
            print(
                f"{Fore.RED}   Check your call graph generation - node keys may not match edge keys."
            )

            # Show more detailed comparison
            all_node_keys = set(self.nodes.keys())
            all_from_keys = set(
                edge["from_key"] for edge in self.edges[:100]
            )  # Sample first 100
            all_to_keys = set(edge["to_key"] for edge in self.edges[:100])

            print(f"{Fore.RED}📊 Debugging key formats:")
            print(f"{Fore.RED}   Sample node keys: {list(all_node_keys)[:3]}")
            print(f"{Fore.RED}   Sample from_keys: {list(all_from_keys)[:3]}")
            print(f"{Fore.RED}   Sample to_keys: {list(all_to_keys)[:3]}")

            # Check if any keys match at all
            from_overlap = all_node_keys & all_from_keys
            to_overlap = all_node_keys & all_to_keys
            print(f"{Fore.RED}   From_key overlap: {len(from_overlap)} keys match")
            print(f"{Fore.RED}   To_key overlap: {len(to_overlap)} keys match")

            raise RuntimeError(
                "CRITICAL: No dependency relationships found - node/edge key mismatch!"
            )

        self.llm = Llm.model_named("ollama:llama3.1")
        print(f"{Fore.BLUE}🤖 LLM initialized: ollama:llama3.1")

        # Initialize ChromaDB for summary embeddings
        chroma_path = Path("code_chromadb")
        chroma_path.mkdir(exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path), settings=Settings(anonymized_telemetry=False)
        )

        # Get or create ChromaDB collection for this repo's summaries
        collection_name = f"code_summaries_{repo_name}"
        try:
            self.chroma_collection = self.chroma_client.get_collection(collection_name)
            print(
                f"{Fore.MAGENTA}🔗 Using existing ChromaDB collection: {collection_name}"
            )
        except:
            self.chroma_collection = self.chroma_client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            print(
                f"{Fore.MAGENTA}🔗 Created new ChromaDB collection: {collection_name}"
            )

    def compute_topology_levels(self) -> Dict[str, int]:
        """Compute dependency levels, detecting and handling cycles"""
        print(f"{Fore.YELLOW}🔄 Computing dependency topology...")
        print(
            f"{Fore.CYAN}📊 Processing {len(self.nodes)} nodes and {len(self.edges)} edges..."
        )

        total_deps = sum(len(deps) for deps in self.dependencies.values())
        print(f"{Fore.CYAN}📊 Total dependency relationships: {total_deps}")

        # Detect cycles using DFS
        print(f"{Fore.BLUE}🔍 Step 1/3: Detecting cycles...")
        visited = set()
        rec_stack = set()
        cycle_edges = set()

        def dfs_cycle_detection(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                try:
                    cycle_start_idx = path.index(node)
                    cycle_edges.add((path[-1], node))
                    if len(cycle_edges) <= 5:
                        print(
                            f"{Fore.YELLOW}⚠️  Cycle detected: {' -> '.join(path[cycle_start_idx:] + [node])}"
                        )
                except ValueError:
                    pass
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.dependencies.get(node, []):
                if dfs_cycle_detection(neighbor, path + [neighbor]):
                    pass

            rec_stack.remove(node)
            return False

        processed_nodes = 0
        for node_key in self.nodes:
            if node_key not in visited:
                dfs_cycle_detection(node_key, [node_key])
            processed_nodes += 1
            if processed_nodes % 5000 == 0:
                print(
                    f"{Fore.CYAN}   📈 Checked {processed_nodes}/{len(self.nodes)} nodes for cycles..."
                )

        if cycle_edges:
            print(
                f"{Fore.RED}🔥 Found {len(cycle_edges)} cycle edges - will ignore them"
            )
        else:
            print(f"{Fore.GREEN}✅ No cycles detected - graph is acyclic!")

        # Build acyclic dependency graph
        print(f"{Fore.BLUE}🔍 Step 2/3: Building acyclic graph...")
        acyclic_deps = defaultdict(list)
        acyclic_dependents = defaultdict(list)  # Corrected: To store callers
        in_degree = defaultdict(int)

        for from_key, to_keys in self.dependencies.items():
            for to_key in to_keys:
                if (from_key, to_key) not in cycle_edges:
                    acyclic_deps[from_key].append(to_key)
                    acyclic_dependents[to_key].append(from_key)  # Corrected
                    in_degree[to_key] += 1

        for node_key in self.nodes:
            if node_key not in in_degree:
                in_degree[node_key] = 0

        # Compute levels using topological sort
        print(f"{Fore.BLUE}🔍 Step 3/3: Computing dependency levels...")
        levels = {}
        queue = deque([node for node, deg in in_degree.items() if deg == 0])

        print(f"{Fore.CYAN}📈 Level 0: {len(queue)} root nodes (no dependencies)")

        for node_key in queue:
            levels[node_key] = 0

        current_level = 0
        processed_count = 0

        while queue:
            current = queue.popleft()
            processed_count += 1

            if processed_count % 5000 == 0:
                print(
                    f"{Fore.CYAN}   📈 Processed {processed_count}/{len(self.nodes)} nodes..."
                )

            # Process nodes that `current` calls
            for neighbor in acyclic_deps.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    # Corrected Logic: Level is 1 + max level of its callers.
                    # By this point, all callers have a level assigned.
                    callers = acyclic_dependents.get(neighbor, [])
                    if callers:
                        max_caller_level = max(levels[caller] for caller in callers)
                        level = max_caller_level + 1
                    else:
                        level = 0  # Should only be for original root nodes

                    levels[neighbor] = level
                    queue.append(neighbor)

                    if level > current_level:
                        print(f"{Fore.CYAN}📈 Discovered Level {level}...")
                        current_level = level

        # Final level statistics
        level_counts = defaultdict(int)
        for level in levels.values():
            level_counts[level] += 1

        max_level = max(levels.values()) if levels else 0
        print(
            f"{Fore.GREEN}✅ Topology computed: {len(levels)} nodes across {max_level + 1} levels"
        )

        print(f"{Fore.CYAN}📊 Level distribution:")
        for level in sorted(level_counts.keys())[:15]:  # Show first 15 levels
            print(f"{Fore.CYAN}   Level {level}: {level_counts[level]} nodes")
        if len(level_counts) > 15:
            print(f"{Fore.CYAN}   ... and {len(level_counts) - 15} more levels")

        return levels

    def extract_code_content(self, file_path: str, start: int, end: int) -> str:
        """Extract code content from file between start and end CHARACTER positions"""
        try:
            full_path = Path(self.repo_path) / file_path

            if not full_path.exists():
                raise FileNotFoundError(f"File does not exist: {full_path}")

            if not full_path.is_file():
                raise ValueError(f"Path is not a file: {full_path}")

            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if len(content) == 0:
                raise ValueError(f"File is empty: {full_path}")

            # Extract by character position
            if start < 0:
                start = 0
            if end > len(content):
                end = len(content)
            if start >= end:
                raise ValueError(
                    f"Invalid character range: start {start} >= end {end} (file length: {len(content)})"
                )

            extracted = content[start:end]

            # Check if content is empty or just whitespace
            stripped_content = extracted.strip()
            if not stripped_content:
                raise ValueError(
                    f"Extracted content is empty or whitespace-only. Raw content: {repr(extracted[:200])}"
                )

            return extracted

        except Exception as e:
            # Only log details when there's an error
            print(f"{Fore.RED}   ❌ CRITICAL: Code extraction failed!")
            print(f"{Fore.RED}      📁 Repo path: {self.repo_path}")
            print(f"{Fore.RED}      📄 File path: {file_path}")
            print(f"{Fore.RED}      📍 Character positions: {start} to {end}")
            print(f"{Fore.RED}      🎯 Full path: {Path(self.repo_path) / file_path}")
            print(
                f"{Fore.RED}      📂 Path exists: {(Path(self.repo_path) / file_path).exists()}"
            )
            if (Path(self.repo_path) / file_path).exists():
                try:
                    with open(
                        Path(self.repo_path) / file_path,
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        file_content = f.read()
                    print(
                        f"{Fore.RED}      📊 Total file length: {len(file_content)} characters"
                    )
                    print(
                        f"{Fore.RED}      📄 File preview (first 200 chars): {repr(file_content[:200])}"
                    )
                except:
                    print(f"{Fore.RED}      📄 Could not read file for debugging")
            print(f"{Fore.RED}      ❌ Error: {e}")
            print(
                f"{Fore.RED}   🛑 TERMINATING: Cannot summarize without code content!"
            )
            raise RuntimeError(
                f"CRITICAL: Failed to extract code content for {file_path} chars {start}-{end}: {e}"
            )

    def create_summary_prompt(
        self,
        node: Dict[str, Any],
        code_content: str,
        dependency_summaries: List[Dict[str, Any]],
    ) -> str:
        """Create LLM prompt for summarizing a code element"""

        deps_text = ""
        if dependency_summaries:
            deps_text = "\n\nDEPENDENCIES:\n"
            for dep in dependency_summaries:
                deps_text += f"- {dep['name']} ({dep['key']})"
                if dep.get("summary"):
                    deps_text += f": {dep['summary']['purpose']}"
                deps_text += "\n"

        return f"""Analyze this {node['kind']} code element and provide a structured summary.

ELEMENT: {node['actual_name']} (in {node['file']})
KIND: {node['kind']}

CODE:
```
{code_content}
```
{deps_text}
Please provide a concise summary focusing on:
1. The primary purpose/responsibility of this code element
2. Key dependencies it relies on (from the dependencies list above)
3. Complexity assessment (Simple/Medium/Complex)
4. Any notable patterns or important details

Be concise but informative. Focus on what this code does, not how it does it."""

    def store_summary_in_chromadb(
        self, node: Dict[str, Any], summary_dict: Dict[str, Any]
    ):
        """Store summary in ChromaDB for semantic search"""
        try:
            # Create document text from summary
            doc_text = summary_dict["purpose"]
            if summary_dict.get("notes"):
                doc_text += f"\n\nNotes: {summary_dict['notes']}"

            # Create metadata
            metadata = {
                "repo_name": self.repo_name,
                "node_key": node["key"],
                "file_path": node.get("file", ""),
                "actual_name": node.get("actual_name", ""),
                "kind": node.get("kind", ""),
                "complexity": summary_dict.get("complexity", "Unknown"),
                "dependency_count": len(summary_dict.get("dependencies", [])),
                "summary_type": "llm_generated",
            }

            # Create unique ID
            chroma_id = f"{self.repo_name}_{node['key']}_summary"

            try:
                self.chroma_collection.add(
                    ids=[chroma_id], documents=[doc_text], metadatas=[metadata]
                )
            except Exception as e:
                # If adding fails (maybe already exists), try updating
                try:
                    self.chroma_collection.update(
                        ids=[chroma_id], documents=[doc_text], metadatas=[metadata]
                    )
                except Exception:
                    # Log but don't fail the whole operation
                    print(f"{Fore.YELLOW}   ⚠️  Could not store in ChromaDB: {e}")

        except Exception as e:
            print(f"{Fore.YELLOW}   ⚠️  ChromaDB error: {e}")

    def summarize_node(self, node: Dict[str, Any]) -> bool:
        """Generate and store summary for a single node"""
        node_key = node["key"]

        print(
            f"{Fore.BLUE}🤖 Calling summarize on node: {node['actual_name']} ({node_key})"
        )

        # Check if already summarized (refresh from database)
        existing = self.db.get_summary(self.repo_id, node_key)
        if existing:
            print(f"{Fore.YELLOW}⏭️  Already summarized - skipping")
            return True

        try:
            # Extract code content - this will raise RuntimeError if it fails
            code_content = self.extract_code_content(
                node["file"], node["start"], node["end"]
            )

            # Get dependencies with their summaries
            dependency_summaries = []
            for dep_key in self.dependencies[node_key]:
                if dep_key in self.nodes:
                    dep_node = self.nodes[dep_key]
                    dep_with_summary = {
                        "key": dep_key,
                        "name": dep_node["name"],
                        "actual_name": dep_node["actual_name"],
                        "kind": dep_node["kind"],
                    }
                    # Get fresh summary from database
                    dep_summary_data = self.db.get_summary(self.repo_id, dep_key)
                    if dep_summary_data:
                        dep_with_summary["summary"] = dep_summary_data["summary"]
                    dependency_summaries.append(dep_with_summary)

            # Create prompt and get LLM summary
            prompt = self.create_summary_prompt(
                node, code_content, dependency_summaries
            )

            print(f"{Fore.CYAN}   ⏳ Calling LLM...")
            start_time = time.time()
            try:
                summary = self.llm.chat_structured(prompt, CodeSummary)
                summary_dict = summary.model_dump()
                llm_time = time.time() - start_time

                # Color code by complexity
                complexity_color = {
                    "Simple": Fore.GREEN,
                    "Medium": Fore.YELLOW,
                    "Complex": Fore.RED,
                }.get(summary_dict.get("complexity", "Unknown"), Fore.WHITE)

                print(
                    f"{Fore.GREEN}✅ {node['actual_name']} "
                    f"{Fore.CYAN}(level {node.get('level', 'N/A')}) "
                    f"{complexity_color}[{summary_dict['complexity']}] "
                    f"{Fore.WHITE}({llm_time:.1f}s)"
                )
                print(f"   {Fore.BLUE}💡 {summary_dict['purpose']}")

            except Exception as llm_error:
                print(f"{Fore.RED}⚠️  LLM error for {node['actual_name']}: {llm_error}")
                # Create fallback summary
                summary_dict = {
                    "purpose": f"Code element {node['actual_name']} of type {node['kind']}",
                    "dependencies": [dep["key"] for dep in dependency_summaries],
                    "complexity": "Unknown",
                    "notes": f"LLM analysis failed: {str(llm_error)}",
                }

            # Store summary immediately (critical for resumability)
            print(f"{Fore.CYAN}   💾 Saving summary to database...")
            self.db.store_summary(self.repo_id, node_key, summary_dict)
            print(f"{Fore.GREEN}   ✅ Saved to database!")

            # Store in ChromaDB for semantic search
            print(f"{Fore.CYAN}   🔗 Storing in ChromaDB...")
            self.store_summary_in_chromadb(node, summary_dict)
            print(f"{Fore.GREEN}   ✅ Saved to ChromaDB!")

            return True

        except RuntimeError as e:
            # Critical errors (like missing code content) should kill the program
            print(f"{Fore.RED}💀 CRITICAL ERROR - TERMINATING PROGRAM")
            print(f"{Fore.RED}Node: {node['actual_name']} ({node_key})")
            print(f"{Fore.RED}Error: {e}")
            raise  # Re-raise to kill the program

        except Exception as e:
            print(f"{Fore.RED}❌ Error summarizing {node_key}: {e}")
            return False

    def get_next_unsummarized_node(
        self, levels: Dict[str, int]
    ) -> Optional[Dict[str, Any]]:
        """Get the next node that needs summarization (lowest level first)"""
        unsummarized = self.db.get_nodes_without_summaries(self.repo_id)

        # Add levels to nodes and sort by level
        nodes_with_levels = []
        for node in unsummarized:
            node_key = node["key"]
            if node_key in levels:
                node["level"] = levels[node_key]
                nodes_with_levels.append(node)

        # Sort by level, then by key for deterministic ordering
        nodes_with_levels.sort(key=lambda n: (n["level"], n["key"]))

        return nodes_with_levels[0] if nodes_with_levels else None

    def process_summaries_continuously(self, levels: Dict[str, int]) -> Dict[str, Any]:
        """Process nodes continuously until all are summarized"""

        # Update node levels in memory
        for node_key, level in levels.items():
            if node_key in self.nodes:
                self.nodes[node_key]["level"] = level

        stats = self.db.get_summary_stats(self.repo_id)
        total_nodes = stats["total_nodes"]

        print(
            f"\n{Back.BLUE}{Fore.WHITE} 🤖 CONTINUOUS SUMMARIZATION + VECTORIZATION {Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}📦 Repository: {self.repo_name}")
        print(f"{Fore.CYAN}📊 Total nodes: {total_nodes}")
        print(f"{Fore.YELLOW}⚡ Running until killed or complete...")
        print(f"{Fore.BLUE}💾 Saves after each summary (resumable)")
        print(f"{Fore.MAGENTA}🔗 Creates vector embeddings in ChromaDB")
        print(f"\n{Fore.GREEN}🚀 Starting to process nodes in dependency order...")

        processed = 0
        start_time = time.time()

        try:
            while True:
                # Get next node to process
                node = self.get_next_unsummarized_node(levels)
                if not node:
                    break

                success = self.summarize_node(node)
                if success:
                    processed += 1

                    # Progress update every 5 nodes
                    if processed % 5 == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        current_stats = self.db.get_summary_stats(self.repo_id)
                        remaining = (
                            current_stats["total_nodes"]
                            - current_stats["summarized_nodes"]
                        )
                        eta = remaining / rate if rate > 0 else 0

                        completion = (
                            current_stats["summarized_nodes"]
                            / current_stats["total_nodes"]
                        ) * 100

                        print(
                            f"\n{Fore.MAGENTA}📈 Progress: {current_stats['summarized_nodes']}/{current_stats['total_nodes']} "
                            f"({completion:.1f}%) "
                            f"Rate: {rate*60:.1f}/min "
                            f"ETA: {eta/60:.1f}min"
                        )

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⏹️  Interrupted by user (Ctrl+C)")

        except Exception as e:
            print(f"\n{Fore.RED}❌ Unexpected error: {e}")

        # Final stats
        final_stats = self.db.get_summary_stats(self.repo_id)
        total_time = time.time() - start_time

        print(f"\n{Back.GREEN}{Fore.WHITE} 📊 FINAL STATUS {Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ Processed: {processed} new summaries")
        print(f"{Fore.BLUE}⏱️  Total time: {total_time/60:.1f} minutes")
        print(
            f"{Fore.CYAN}📊 Final: {final_stats['summarized_nodes']}/{final_stats['total_nodes']} nodes summarized"
        )
        print(f"{Fore.MAGENTA}🔗 ChromaDB: {processed} new summary embeddings stored")

        if final_stats["total_nodes"] > 0:
            completion = (
                final_stats["summarized_nodes"] / final_stats["total_nodes"]
            ) * 100
            if completion == 100:
                print(
                    f"{Fore.GREEN}🎉 Repository completely summarized and vectorized!"
                )
            else:
                print(
                    f"{Fore.YELLOW}💾 Saved progress - restart to continue from {completion:.1f}%"
                )
        else:
            print(f"{Fore.YELLOW}⚠️ No nodes in repository to process.")

        if processed > 0:
            avg_time = total_time / processed
            print(f"{Fore.YELLOW}⚡ Average: {avg_time:.1f}s per summary")

        return {
            "processed": processed,
            "total": final_stats["total_nodes"],
            "final_summarized": final_stats["summarized_nodes"],
            "time_taken": total_time,
        }


def find_repository_to_process(db: Database) -> Optional[Dict[str, Any]]:
    """Find a repository that needs summarization"""
    cur = db.conn.cursor()
    cur.execute(
        """
        SELECT r.id, r.name, r.path, COUNT(n.key) as node_count, COUNT(s.node_key) as summary_count
        FROM repos r
        LEFT JOIN nodes n ON r.id = n.repo_id
        LEFT JOIN summaries s ON n.key = s.node_key AND n.repo_id = s.repo_id
        GROUP BY r.id, r.name, r.path
        HAVING node_count > 0
        ORDER BY (CAST(summary_count AS REAL) / CAST(node_count AS REAL)) ASC, node_count DESC
        LIMIT 1
    """
    )

    row = cur.fetchone()
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "path": row[2],
            "node_count": row[3],
            "summary_count": row[4],
        }
    return None


def main():
    print(f"{Back.MAGENTA}{Fore.WHITE} 🚀 CODE SUMMARIZER STARTING {Style.RESET_ALL}")

    db_path = Path("code_callgraph.sqlite")
    if not db_path.exists():
        print(f"{Fore.RED}❌ Database file not found: {db_path}")
        print(
            f"{Fore.YELLOW}💡 Make sure code_callgraph.sqlite exists in current directory"
        )
        print(f"{Fore.CYAN}🔍 Looking at: {db_path.absolute()}")
        sys.exit(1)

    db = Database(db_path)

    # Find repository to process
    repo = find_repository_to_process(db)
    if not repo:
        print(f"{Fore.RED}❌ No repositories found with nodes to summarize")
        sys.exit(1)

    completion = (
        (repo["summary_count"] / repo["node_count"]) * 100
        if repo["node_count"] > 0
        else 0
    )
    print(f"{Fore.CYAN}📦 Processing repository: {Fore.WHITE}{repo['name']}")
    print(f"{Fore.CYAN}📁 Path: {Fore.WHITE}{repo['path']}")
    print(
        f"{Fore.CYAN}📊 Current progress: {Fore.WHITE}{repo['summary_count']}/{repo['node_count']} ({completion:.1f}%)"
    )

    # Initialize processor and run
    processor = CallGraphProcessor(db, repo["id"], repo["name"], repo["path"])

    # Compute topology if needed
    print(f"\n{Back.YELLOW}{Fore.BLACK} 🔄 TOPOLOGY ANALYSIS {Style.RESET_ALL}")
    levels = processor.compute_topology_levels()
    print(f"{Back.GREEN}{Fore.WHITE} ✅ FINISHED CALCULATING LEVELS {Style.RESET_ALL}")

    # Process summaries continuously
    result = processor.process_summaries_continuously(levels)


if __name__ == "__main__":
    main()
