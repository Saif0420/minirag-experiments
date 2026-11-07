import os
from minirag import MiniRAG, QueryParam
from minirag.llm.ollama import ollama_model_complete, ollama_embed
from minirag.utils import EmbeddingFunc

WORKING_DIR = "./minirag_cache"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.2",
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(texts, embed_model="nomic-embed-text")
    )
)

from pathlib import Path

dataset_dir = Path("./data/lihua_world")
files = list(dataset_dir.rglob("*.txt"))[:25]

print(f"Found {len(files)} files")

texts = [f.read_text(encoding="utf-8") for f in files]

print("Indexing dataset...")
rag.insert(texts)
print("Indexing complete.")