import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_graphml("./minirag_cache_qwen/graph_chunk_entity_relation.graphml")

degrees = dict(G.degree())
top = sorted(degrees, key=degrees.get, reverse=True)[:15]

pos = nx.spring_layout(G, seed=42, k=0.5)

plt.figure(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, node_size=[50 + degrees[n] * 40 for n in G], node_color="#4f46e5", alpha=0.7)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, labels={n: n.strip('"') for n in top}, font_size=9)

plt.title("MiniRAG knowledge graph, 25 LiHua-World files, built with Qwen2.5-3B")
plt.axis("off")
plt.savefig("results/graph_qwen.png", dpi=150, bbox_inches="tight")
print("Saved to results/graph_qwen.png")