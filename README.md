# MiniRAG Experiments

Reproducing and extending [MiniRAG](https://github.com/HKUDS/MiniRAG) (ACL 2026), a graph-based retrieval-augmented generation (RAG) framework built for small language models, using only local models through Ollama.

## Summary

- Reproduced MiniRAG on a 25-file subset of the LiHua-World dataset and compared it with standard vector-based (naive) RAG on 12 hand-written questions.
- In the first experiment, MiniRAG did **not** beat naive RAG, the opposite of the paper's result. The knowledge graph built by llama3.2 was very sparse (218 nodes, 41 edges).
- Rebuilding the graph with Qwen2.5-3B produced a much denser graph (124 nodes, 111 edges). MiniRAG went from 0 to 3 correct answers, while naive RAG, which doesn't use the graph, stayed the same.
- All of MiniRAG's correct answers were questions that need information from several conversations, which is what the graph is designed for.
- Conclusion: in this setup, MiniRAG's advantage depends heavily on the quality of the knowledge graph, and with small local models that graph is the weak point.

## What MiniRAG does

Standard RAG splits documents into chunks and retrieves the ones most similar to the question. MiniRAG also builds a knowledge graph of entities (people, places, topics) and the relationships between them, then uses that graph structure to find relevant information. The idea is that following the graph helps small models answer questions that need several pieces of information connected together.

## Setup

Requirements: Python 3.12, [Ollama](https://ollama.com), macOS or Linux.

Python 3.14 could not build some dependencies (graspologic/gensim), and the PyPI package `minirag-hku` had an import bug, so MiniRAG is installed directly from GitHub.

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install git+https://github.com/HKUDS/MiniRAG.git
python -m pip install python-dotenv sentence-transformers ollama networkx matplotlib
```

Pull the models:

```bash
ollama pull qwen2.5:3b
ollama pull llama3.2
ollama pull nomic-embed-text
```

The LiHua-World dataset comes from the MiniRAG repository and goes in `data/lihua_world/`.

## How to run

```bash
python src/index.py
python src/evaluate.py
python src/visualize.py
```

- `index.py` builds the knowledge graph from the first 25 files. It is currently set to Qwen2.5-3B. To reproduce experiment 1, set the model to `llama3.2` and the folder to `./minirag_cache`.
- `evaluate.py` asks the 12 questions in both MiniRAG and naive mode and saves the answers to `results/`.
- `visualize.py` draws the knowledge graph.
- `query.py` asks a single question in both modes, which is useful for quick tests.

Indexing 25 files with a local model takes around two hours on a MacBook.

## Project structure

```
minirag-experiments/
├── data/lihua_world/       LiHua-World dataset
├── results/
│   ├── comparison.json      experiment 1 answers
│   ├── comparison_qwen.json experiment 2 answers
│   └── graph_qwen.png       knowledge graph
├── src/
│   ├── index.py
│   ├── query.py
│   ├── evaluate.py
│   └── visualize.py
└── requirements.txt
```

## Evaluation

12 questions written from the 25 indexed files, each with a known answer:

- 7 single-fact questions, answered in one conversation.
- 5 connection questions, which need information from several conversations.

Each answer was scored by hand as correct, partial or wrong. Full answers are in `results/`.

## Experiment 1: graph built with llama3.2

Indexing with llama3.2, querying with Qwen2.5-3B, embeddings with nomic-embed-text.

llama3.2 could not follow MiniRAG's keyword extraction format during querying, so Qwen2.5-3B was used for that step.

| Method | Correct | Partial | Wrong |
|---|---|---|---|
| MiniRAG | 0 | 1 | 11 |
| Naive RAG | 1 | 2 | 9 |

- MiniRAG's answers kept mentioning the same topics (rent, workouts, the neighborhood), whatever the question.
- Naive RAG sometimes retrieved the right fact but used it in the wrong answer.
- The graph was very sparse: 218 nodes but only 41 edges.

## Experiment 2: graph built with Qwen2.5-3B

Same files, same questions. Only the indexing model changed.

| Indexing model | Nodes | Edges |
|---|---|---|
| llama3.2 | 218 | 41 |
| qwen2.5:3b | 124 | 111 |

| Experiment | MiniRAG (correct / partial / wrong) | Naive RAG (correct / partial / wrong) |
|---|---|---|
| 1: llama3.2 graph | 0 / 1 / 11 | 1 / 2 / 9 |
| 2: qwen2.5 graph | 3 / 1 / 8 | 1 / 2 / 9 |

- Naive RAG doesn't use the graph and its score didn't change, so it acts as a control. MiniRAG improved, which suggests the denser graph caused the gain.
- All three of MiniRAG's correct answers were connection questions.
- Single-fact questions failed for both methods in both experiments (0 of 7).
- Naive RAG sometimes made up content, for example inventing a whole conversation for one question.

![Knowledge graph](results/graph_qwen.png)

- LiHua is the central hub, connected to the people he talks to most (Wolfgang Schulz, Adam Smith, Jake Watson, ChaeSong-hwa).
- The graph separates LiHua's own conversations from an unrelated group chat about Game of Thrones.
- "LIHUA" and "LI HUA" were extracted as two different entities, so LiHua's connections are split across two nodes. This may explain part of the retrieval failures.
- Many entities have no connections at all (the outer ring), even with the denser graph.

## Limitations

- Only 25 files from the dataset, because indexing with a local model on a laptop is slow.
- 12 questions scored by hand. The results show a direction, not a conclusive result.
- Small local models, not the exact setup from the paper.

## Future work

- Merge duplicate entities such as "LIHUA" and "LI HUA" and test whether single-fact questions improve.
- Run on more files with more questions.
- Test on a different domain, such as technical documents, to see if the results hold outside chat logs.

## Credits

MiniRAG and the LiHua-World dataset are by the HKU Data Intelligence Lab: [github.com/HKUDS/MiniRAG](https://github.com/HKUDS/MiniRAG). This repository only reproduces and extends their work.