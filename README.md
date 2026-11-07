# MiniRAG Experiments

Reproducing and extending MiniRAG (ACL 2026), a graph-based RAG framework, using local LLMs through Ollama.

## Setup notes

- Python 3.14 couldn't build some dependencies (graspologic/gensim), so the project runs on Python 3.12.
- The PyPI package `minirag-hku` had an import bug, so I installed MiniRAG directly from GitHub.

## Findings

- Indexed a 5-file subset of LiHua-World: 45 nodes, 63 edges.
- llama3.2 couldn't follow MiniRAG's keyword extraction format in "mini" query mode, so querying uses Qwen2.5-3B.
- First query ("What does LiHua do for work?"): MiniRAG found LiHua's band activity through the graph; naive RAG found nothing relevant. Neither found a job, since the 5-file subset doesn't mention one.
- Indexed 25 files: 218 nodes but only 41 edges. llama3.2 extracted many entities but few relationships, which gives MiniRAG a sparse graph to work with.
