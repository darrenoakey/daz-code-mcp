#!/usr/bin/env python3
"""
Git Repository Code Scanner with ChromaDB Vector Database Indexing
Memory-optimized version with batch processing and garbage collection
"""

import os
import sys
import time
import hashlib
import logging
import subprocess
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import json

import chromadb
from chromadb.config import Settings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Test tree-sitter imports with detailed error reporting
def setup_tree_sitter():
    """Setup tree-sitter with proper error handling"""
    try:
        from tree_sitter import Parser, Language
    except ImportError as e:
        logger.error(f"Failed to import tree-sitter: {e}")
        logger.error("Install with: pip install tree-sitter")
        sys.exit(1)

    languages = {}

    # Language configurations with their actual function names
    language_configs = [
        ("python", "tree_sitter_python", "language"),
        ("javascript", "tree_sitter_javascript", "language"),
        ("typescript", "tree_sitter_typescript", "language_typescript"),
        ("tsx", "tree_sitter_typescript", "language_tsx"),  # For TSX files
        ("java", "tree_sitter_java", "language"),
        ("cpp", "tree_sitter_cpp", "language"),
        ("go", "tree_sitter_go", "language"),
        ("rust", "tree_sitter_rust", "language"),
    ]

    failed_languages = []

    for lang_name, module_name, function_name in language_configs:
        try:
            module = __import__(module_name)
            lang_func = getattr(module, function_name)
            language_capsule = lang_func()
            # Wrap the PyCapsule in a Language object
            language = Language(language_capsule)
            languages[lang_name] = language
            logger.info(f"✓ Loaded {lang_name} language support")
        except ImportError:
            failed_languages.append(
                f"{module_name} (install with: pip install {module_name})"
            )
        except AttributeError as e:
            logger.error(f"✗ {module_name} has no {function_name}: {e}")
            failed_languages.append(f"{module_name} (API mismatch)")
        except Exception as e:
            logger.error(f"✗ Error loading {lang_name}: {e}")
            failed_languages.append(f"{module_name} (error: {e})")

    if failed_languages:
        logger.error("Failed to load the following language parsers:")
        for lang in failed_languages:
            logger.error(f"  - {lang}")
        logger.error("Tree-sitter code parsing is required for this application.")
        sys.exit(1)

    if not languages:
        logger.error("No tree-sitter languages could be loaded. Cannot continue.")
        sys.exit(1)

    logger.info(f"Successfully loaded {len(languages)} language parsers")
    return languages


# Load tree-sitter languages
LANGUAGES = setup_tree_sitter()

# Constants - reduced for memory efficiency
CHUNK_SIZE = 512  # Reduced from 1024
OVERLAP_SIZE = 128  # Reduced from 256
STEP_SIZE = CHUNK_SIZE - OVERLAP_SIZE
COLLECTION_NAME = "code_chunks"
METADATA_FILE = ".code_indexer_metadata.json"
BATCH_SIZE = 50  # Process files in batches
MAX_FILE_SIZE = 1024 * 1024  # Skip files larger than 1MB to save memory

# Language mappings - updated to include tsx
LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",  # Now maps to tsx language
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
}


def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil

    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0


def get_git_files(repo_path: Path) -> Set[str]:
    """Get list of files tracked by git"""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return (
            set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting git files: {e}")
        return set()


def calculate_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of file content"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {filepath}: {e}")
        return ""


def chunk_content(content: str) -> List[Tuple[str, int, int]]:
    """Break content into overlapping chunks"""
    chunks = []
    content_bytes = content.encode("utf-8")
    pos = 0

    while pos < len(content_bytes):
        end_pos = min(pos + CHUNK_SIZE, len(content_bytes))
        chunk_bytes = content_bytes[pos:end_pos]

        # Ensure we don't cut in the middle of a UTF-8 character
        while end_pos > pos:
            try:
                chunk_text = chunk_bytes.decode("utf-8")
                break
            except UnicodeDecodeError:
                end_pos -= 1
                chunk_bytes = content_bytes[pos:end_pos]

        if end_pos > pos:
            chunks.append((chunk_text, pos, end_pos))

        pos += STEP_SIZE

    return chunks


def extract_code_elements(content: str, language: str, filepath: str) -> List[Dict]:
    """Extract functions and classes from code using tree-sitter"""
    if language not in LANGUAGES:
        return []

    try:
        from tree_sitter import Parser, Language

        # In newer tree-sitter versions, pass language directly to Parser constructor
        parser = Parser(LANGUAGES[language])
        tree = parser.parse(bytes(content, "utf8"))

        elements = []

        def traverse_node(node, depth=0):
            # Limit recursion depth to avoid stack overflow on large files
            if depth > 50:
                return

            # Extract functions
            if node.type in [
                "function_definition",
                "function_declaration",
                "method_definition",
                "arrow_function",
                "function_expression",
                "method_signature",
                "constructor_definition",
                "function",
                "func_declaration",
            ]:
                name_node = None
                for child in node.children:
                    if child.type in ["identifier", "name", "property_identifier"]:
                        name_node = child
                        break

                if name_node:
                    func_name = content[name_node.start_byte : name_node.end_byte]
                    func_text = content[node.start_byte : node.end_byte]

                    # Limit function text size to save memory
                    if len(func_text) > 10000:
                        func_text = func_text[:10000] + "... [truncated]"

                    elements.append(
                        {
                            "type": "function",
                            "name": func_name,
                            "text": func_text,
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "file_path": filepath,
                        }
                    )

            # Extract classes
            elif node.type in [
                "class_definition",
                "class_declaration",
                "interface_declaration",
                "type_alias_declaration",
                "struct_item",
                "impl_item",
            ]:
                name_node = None
                for child in node.children:
                    if child.type in ["identifier", "name", "type_identifier"]:
                        name_node = child
                        break

                if name_node:
                    class_name = content[name_node.start_byte : name_node.end_byte]
                    class_text = content[node.start_byte : node.end_byte]

                    # Limit class text size to save memory
                    if len(class_text) > 10000:
                        class_text = class_text[:10000] + "... [truncated]"

                    elements.append(
                        {
                            "type": "class",
                            "name": class_name,
                            "text": class_text,
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "file_path": filepath,
                        }
                    )

            # Recursively traverse children
            for child in node.children:
                traverse_node(child, depth + 1)

        traverse_node(tree.root_node)
        return elements

    except Exception as e:
        logger.error(f"Error parsing {filepath} with tree-sitter: {e}")
        return []


class CodeRepositoryIndexer:
    def __init__(self, repo_path: Path, db_path: Path, repo_name: str):
        self.repo_path = repo_path
        self.db_path = db_path
        self.repo_name = repo_name
        self.collection_name = f"code_chunks_{repo_name}"
        self.metadata_file = db_path / f".code_indexer_metadata_{repo_name}.json"

        # Verify this is a git repository
        if not (repo_path / ".git").exists():
            raise ValueError(f"Path {repo_path} is not a git repository")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")

        # Load metadata
        self.file_hashes = self.load_metadata()

        # Get git files
        self.git_files = get_git_files(repo_path)
        logger.info(f"Found {len(self.git_files)} files in git repository")

    def load_metadata(self) -> Dict[str, str]:
        """Load file hashes from metadata file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        return {}

    def save_metadata(self):
        """Save file hashes to metadata file"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.file_hashes, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            # Clear metadata file
            self.file_hashes = {}
            self.save_metadata()
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def remove_file_chunks(self, filepath: str):
        """Remove all chunks for a given file from the database"""
        try:
            results = self.collection.get(where={"file_path": filepath})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Removed {len(results['ids'])} chunks for {filepath}")
        except Exception as e:
            logger.error(f"Error removing chunks for {filepath}: {e}")

    def index_file(self, filepath: Path):
        """Index a single file with memory optimization"""
        relative_path = str(filepath.relative_to(self.repo_path))

        # Skip if not in git
        if relative_path not in self.git_files:
            return

        try:
            # Check file size first
            file_size = filepath.stat().st_size
            if file_size > MAX_FILE_SIZE:
                logger.warning(
                    f"Skipping large file {relative_path} ({file_size} bytes)"
                )
                return

            # Read file content
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Skip very large files in memory too
            if len(content) > MAX_FILE_SIZE:
                logger.warning(f"Skipping large content file {relative_path}")
                return

            # Calculate hash
            current_hash = calculate_file_hash(filepath)

            # Check if file has changed
            if (
                relative_path in self.file_hashes
                and self.file_hashes[relative_path] == current_hash
            ):
                logger.debug(f"File unchanged: {relative_path}")
                return

            # Remove old chunks if file existed before
            if relative_path in self.file_hashes:
                self.remove_file_chunks(relative_path)

            # Prepare data for ChromaDB in smaller batches
            all_ids = []
            all_documents = []
            all_metadatas = []

            # Add filename chunk
            filename_id = f"{self.repo_name}_{relative_path}_filename"
            all_ids.append(filename_id)
            # Truncate content preview to save memory
            content_preview = content[:1000] + "..." if len(content) > 1000 else content
            all_documents.append(f"{filepath.name}\n{content_preview}")
            all_metadatas.append(
                {
                    "file_path": relative_path,
                    "repo_name": self.repo_name,
                    "chunk_type": "filename",
                    "element_name": filepath.name,
                    "full_path": str(filepath),
                }
            )

            # Regular content chunks (fewer chunks to save memory)
            content_chunks = chunk_content(content)
            # Limit number of chunks per file
            max_chunks = min(len(content_chunks), 20)
            for i, (chunk_text, start_pos, end_pos) in enumerate(
                content_chunks[:max_chunks]
            ):
                chunk_id = f"{self.repo_name}_{relative_path}_chunk_{i}"
                all_ids.append(chunk_id)
                all_documents.append(chunk_text)
                all_metadatas.append(
                    {
                        "file_path": relative_path,
                        "repo_name": self.repo_name,
                        "chunk_type": "content",
                        "chunk_index": i,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "full_path": str(filepath),
                    }
                )

            # Code element chunks (functions and classes) - limit to save memory
            file_extension = filepath.suffix.lower()
            if file_extension in LANGUAGE_EXTENSIONS:
                language = LANGUAGE_EXTENSIONS[file_extension]
                code_elements = extract_code_elements(content, language, relative_path)

                # Limit number of code elements to save memory
                max_elements = min(len(code_elements), 50)
                for element in code_elements[:max_elements]:
                    # Make ID unique by including start_byte position
                    element_id = f"{self.repo_name}_{relative_path}_{element['type']}_{element['name']}_{element['start_byte']}"
                    all_ids.append(element_id)
                    all_documents.append(element["text"])
                    all_metadatas.append(
                        {
                            "file_path": relative_path,
                            "repo_name": self.repo_name,
                            "chunk_type": element["type"],
                            "element_name": element["name"],
                            "language": language,
                            "start_byte": element["start_byte"],
                            "end_byte": element["end_byte"],
                            "full_path": str(filepath),
                        }
                    )

            # Add to ChromaDB in smaller batches
            if all_ids:
                # Process in small batches to avoid memory issues
                batch_size = 100
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i : i + batch_size]
                    batch_docs = all_documents[i : i + batch_size]
                    batch_meta = all_metadatas[i : i + batch_size]

                    self.collection.add(
                        ids=batch_ids, documents=batch_docs, metadatas=batch_meta
                    )

                # Update hash
                self.file_hashes[relative_path] = current_hash
                self.save_metadata()

                logger.info(f"Indexed {len(all_ids)} chunks for {relative_path}")

            # Clear variables to free memory
            del content, all_ids, all_documents, all_metadatas
            if "code_elements" in locals():
                del code_elements
            if "content_chunks" in locals():
                del content_chunks

        except Exception as e:
            logger.error(f"Error indexing {filepath}: {e}")

    def remove_file(self, filepath: Path):
        """Remove a file from the index"""
        try:
            relative_path = str(filepath.relative_to(self.repo_path))
            self.remove_file_chunks(relative_path)

            if relative_path in self.file_hashes:
                del self.file_hashes[relative_path]
                self.save_metadata()

            logger.info(f"Removed {relative_path} from index")
        except Exception as e:
            logger.error(f"Error removing {filepath}: {e}")

    def update_git_files(self):
        """Refresh the list of git-tracked files"""
        self.git_files = get_git_files(self.repo_path)

    def initial_index(self):
        """Perform initial indexing with memory management"""
        logger.info("Starting initial index with memory optimization...")

        git_files_list = list(self.git_files)
        total_files = len(git_files_list)
        indexed_count = 0

        # Process files in batches
        for i in range(0, total_files, BATCH_SIZE):
            batch_files = git_files_list[i : i + BATCH_SIZE]

            for relative_path in batch_files:
                filepath = self.repo_path / relative_path
                if filepath.exists() and filepath.is_file():
                    self.index_file(filepath)
                    indexed_count += 1

                    if indexed_count % 100 == 0:
                        memory_mb = get_memory_usage()
                        logger.info(
                            f"Progress: {indexed_count}/{total_files} files indexed, Memory: {memory_mb:.1f}MB"
                        )

            # Force garbage collection after each batch
            gc.collect()

            # Log memory usage
            memory_mb = get_memory_usage()
            logger.info(
                f"Batch complete. {indexed_count}/{total_files} files, Memory: {memory_mb:.1f}MB"
            )

            # If memory usage is too high, warn and consider pausing
            if memory_mb > 2000:  # 2GB threshold
                logger.warning(
                    f"High memory usage: {memory_mb:.1f}MB. Consider reducing batch size."
                )

        logger.info(f"Initial index complete. Indexed {indexed_count} files.")


class CodeRepositoryEventHandler(FileSystemEventHandler):
    def __init__(self, indexer: CodeRepositoryIndexer):
        self.indexer = indexer

    def on_created(self, event):
        if not event.is_directory:
            filepath = Path(event.src_path)
            try:
                relative_path = str(filepath.relative_to(self.indexer.repo_path))
                if relative_path in self.indexer.git_files:
                    logger.info(f"File created: {event.src_path}")
                    self.indexer.index_file(filepath)
            except ValueError:
                pass

    def on_modified(self, event):
        if not event.is_directory:
            filepath = Path(event.src_path)
            try:
                relative_path = str(filepath.relative_to(self.indexer.repo_path))
                if relative_path in self.indexer.git_files:
                    logger.info(f"File modified: {event.src_path}")
                    self.indexer.index_file(filepath)
            except ValueError:
                # File is outside repo path
                pass

    def on_deleted(self, event):
        if not event.is_directory:
            filepath = Path(event.src_path)
            try:
                logger.info(f"File deleted: {event.src_path}")
                self.indexer.remove_file(filepath)
            except ValueError:
                # File is outside repo path
                pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python code_scanner.py <repo_path> [repo_name]")
        sys.exit(1)

    repo_path = Path(sys.argv[1]).resolve()
    repo_name = sys.argv[2] if len(sys.argv) > 2 else repo_path.name

    if not repo_path.exists():
        print(f"Error: Repository path {repo_path} does not exist")
        sys.exit(1)

    # Set up database in script directory
    script_dir = Path(__file__).parent
    db_path = script_dir / "code_chromadb"
    db_path.mkdir(exist_ok=True)

    # Create indexer
    try:
        indexer = CodeRepositoryIndexer(repo_path, db_path, repo_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Perform initial indexing
    indexer.initial_index()

    # Set up file watcher
    event_handler = CodeRepositoryEventHandler(indexer)
    observer = Observer()
    observer.schedule(event_handler, str(repo_path), recursive=True)

    # Start watching
    observer.start()
    logger.info(f"File watcher started for {repo_path}. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(5)  # Light sleep to reduce CPU usage
            # Periodically refresh git files list and run garbage collection
            if time.time() % 60 < 5:  # Every minute or so
                indexer.update_git_files()
                gc.collect()
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopping file watcher...")

    observer.join()
    logger.info("File watcher stopped.")


if __name__ == "__main__":
    main()
