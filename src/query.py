from minirag import MiniRAG, QueryParam
from minirag.llm.ollama import ollama_model_complete, ollama_embed
from minirag.utils import EmbeddingFunc

rag = MiniRAG(
    working_dir="./minirag_cache",
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.2",
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(texts, embed_model="nomic-embed-text")
    )
)

question = "What does LiHua do for work?"

print("MiniRAG answer:")
print(rag.query(question, param=QueryParam(mode="mini")))

print("\nNaive RAG answer:")
print(rag.query(question, param=QueryParam(mode="naive")))