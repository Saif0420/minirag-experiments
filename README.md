# MiniRAG Experiments

Reproducing and extending MiniRAG (ACL 2026), a graph-based RAG framework, using local LLMs through Ollama.

## Setup notes

- Python 3.14 couldn't build some dependencies (graspologic/gensim), so the project runs on Python 3.12.
- The PyPI package `minirag-hku` had an import bug, so I installed MiniRAG directly from GitHub.

## Findings

- Indexed a 5-file subset of LiHua-World: 45 nodes, 63 edges.
- llama3.2 couldn't follow MiniRAG's keyword extraction format in "mini" query mode, so querying uses Qwen2.5-3B.
