import json
from minirag import MiniRAG, QueryParam
from minirag.llm.ollama import ollama_model_complete, ollama_embed
from minirag.utils import EmbeddingFunc

rag = MiniRAG(
    working_dir="./minirag_cache_qwen",
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:3b",
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(texts, embed_model="nomic-embed-text")
    )
)

questions = [
    {"q": "Which game was LiHua thinking about buying?", "expected": "Sekiro: Shadows Die Twice"},
    {"q": "What did Adam Smith ask LiHua to measure, and why?", "expected": "The basement window, to get curtains made"},
    {"q": "What did LiHua and Wolfgang plan for dinner, and at what time?", "expected": "Sichuan hot pot at 7 pm"},
    {"q": "Who does Jake Watson name as one of the best defenders in FC Barcelona's history?", "expected": "Carles Puyol"},
    {"q": "What event is Hailey Johnson planning at the bakery?", "expected": "A bread-tasting event, with LiHua helping gather customer feedback"},
    {"q": "What is LiHua's favorite first-person shooter?", "expected": "Call of Duty: Warzone"},
    {"q": "How did LiHua pay his rent to Adam?", "expected": "Bank transfer"},
    {"q": "Who does LiHua play music in a band with and also go out to eat or drink with?", "expected": "Wolfgang Schulz"},
    {"q": "What different things has Adam Smith asked LiHua about?", "expected": "Measuring the basement window, choosing an air conditioner, and the rent"},
    {"q": "What ideas have band members suggested to improve their music?", "expected": "Binaural tones, original songs, improvisational solos"},
    {"q": "What style of music does LiHua's band play?", "expected": "Jazz fusion"},
    {"q": "Who supports LiHua's fitness training, and what advice did they give?", "expected": "Jennifer Moore: bench press progress and protein after workouts"},
]

results = []

for item in questions:
    print(f"Asking: {item['q']}")
    results.append({
        "question": item["q"],
        "expected": item["expected"],
        "minirag": rag.query(item["q"], param=QueryParam(mode="mini")),
        "naive": rag.query(item["q"], param=QueryParam(mode="naive")),
    })

with open("results/comparison_qwen.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved to results/comparison_qwen.json")