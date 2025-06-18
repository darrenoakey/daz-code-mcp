# Code Indexer and Search

[![Banner Image](banner.jpg)](banner.jpg)

## Purpose

This project provides tools to index code repositories and search through them efficiently. It allows you to:

*   Index code from git repositories.
*   Search code snippets based on keywords.
*   Search for specific code elements like functions and classes.
*   Search for files by filename.
*   Retrieve full file contents based on a search query.

## Installation

1.  Ensure you have Python 3.7 or higher installed.
2.  Clone the repository.
3.  Install the required Python packages:

    ```bash
    pip install chromadb watchdog tree_sitter mcp
    pip install tree_sitter_python tree_sitter_javascript tree_sitter_typescript tree_sitter_java tree_sitter_cpp tree_sitter_go tree_sitter_rust
    ```

## Usage

### 1. Indexing Code

Use `code-scan.py` to index a git repository. This script scans the repository, extracts code elements, and stores them in a ChromaDB database.

```bash
python code-scan.py <repo_path> [repo_name]
```

*   `<repo_path>`: The path to the git repository you want to index.
*   `[repo_name]`: An optional name for the repository in the index. If not provided, the repository's directory name will be used.

**Example:**

```bash
python code-scan.py /path/to/my/repo my_repo
```

### 2. Searching Code

Use `daz-code-mcp.py` to search the indexed code. This script starts an MCP (Model Context Protocol) server that allows you to perform various search queries.

1.  Run the MCP server:

    ```bash
    python daz-code-mcp.py
    ```

2.  Use an MCP client to interact with the server and perform searches. The server provides the following tools:

    *   `search_code_snippets`: Search for relevant code chunks.
    *   `search_functions`: Search specifically for functions.
    *   `search_classes`: Search specifically for classes.
    *   `search_files`: Search for files by filename.
    *   `search_code_full`: Search for relevant code and retrieve the entire file content.

**Example (MCP Client Interaction):**

Here's how you might interact with the server using a hypothetical MCP client:

```python
# Assumes you have an MCP client set up and connected

# Search for code snippets related to 'authentication'
result = mcp_client.call_tool(
    "search_code_snippets", {"query": "authentication", "limit": 5}
)
print(result)

# Search for functions related to 'database connection'
result = mcp_client.call_tool(
    "search_functions", {"query": "database connection", "limit": 3}
)
print(result)

# Search for files with 'user_profile' in the filename
result = mcp_client.call_tool(
    "search_files", {"query": "user_profile", "limit": 2}
)
print(result)
```

**Note:**  This example assumes you have an MCP client library and the `mcp_client` object is properly initialized and connected to the `daz-code-mcp.py` server.
