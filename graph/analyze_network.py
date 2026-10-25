import os
import pandas as pd
import networkx as nx

PROJECT = r"C:\Users\adity\OneDrive\Desktop\AI-Pancreatic-Cancer-Biomarker"

NODES_FILE = os.path.join(
    PROJECT,
    "data",
    "interactions",
    "nodes.csv"
)

EDGES_FILE = os.path.join(
    PROJECT,
    "data",
    "interactions",
    "edges.csv"
)

OUT_FILE = os.path.join(
    PROJECT,
    "data",
    "interactions",
    "node_centrality.csv"
)

print("Loading network...")

nodes = pd.read_csv(NODES_FILE)
edges = pd.read_csv(EDGES_FILE)

print("Nodes:", len(nodes))
print("Edges:", len(edges))
print()

# --------------------------------
# Create undirected RNA network
# --------------------------------

G = nx.Graph()

for _, row in nodes.iterrows():
    G.add_node(
        row["node_id"],
        rna_id=row["rna_id"],
        rna_type=row["rna_type"]
    )

for _, row in edges.iterrows():

    source = row["source"]
    target = row["target"]

    if source in G and target in G:
        G.add_edge(
            source,
            target
        )

print("Network created")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print()

# --------------------------------
# Degree centrality
# --------------------------------

print("Calculating degree centrality...")

degree = dict(G.degree())

# --------------------------------
# Betweenness centrality
# --------------------------------

print("Calculating betweenness centrality...")

betweenness = nx.betweenness_centrality(
    G,
    normalized=True
)

# --------------------------------
# Closeness centrality
# --------------------------------

print("Calculating closeness centrality...")

closeness = nx.closeness_centrality(G)

# --------------------------------
# PageRank
# --------------------------------

print("Calculating PageRank...")

pagerank = nx.pagerank(G)

# --------------------------------
# Create results table
# --------------------------------

results = []

for node in G.nodes():

    results.append({
        "node_id": node,
        "rna_id": G.nodes[node].get("rna_id", ""),
        "rna_type": G.nodes[node].get("rna_type", ""),
        "degree": degree.get(node, 0),
        "betweenness": betweenness.get(node, 0),
        "closeness": closeness.get(node, 0),
        "pagerank": pagerank.get(node, 0)
    })

result = pd.DataFrame(results)

# --------------------------------
# Rank nodes
# --------------------------------

result["degree_rank"] = (
    result["degree"]
    .rank(
        ascending=False,
        method="min"
    )
)

result["betweenness_rank"] = (
    result["betweenness"]
    .rank(
        ascending=False,
        method="min"
    )
)

result["pagerank_rank"] = (
    result["pagerank"]
    .rank(
        ascending=False,
        method="min"
    )
)

# Sort by degree
result = result.sort_values(
    [
        "degree",
        "betweenness",
        "pagerank"
    ],
    ascending=False
)

# --------------------------------
# Save
# --------------------------------

result.to_csv(
    OUT_FILE,
    index=False
)

print()
print("======================================")
print("NETWORK ANALYSIS COMPLETED")
print("======================================")

print()
print("Output:")
print(OUT_FILE)

print()
print("Top 20 network hubs:")

print(
    result[
        [
            "node_id",
            "rna_id",
            "rna_type",
            "degree",
            "betweenness",
            "pagerank"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print()
print("Top lncRNA hubs:")

print(
    result[
        result["rna_type"] == "lncRNA"
    ][
        [
            "rna_id",
            "degree",
            "betweenness",
            "pagerank"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("Top miRNA hubs:")

print(
    result[
        result["rna_type"] == "miRNA"
    ][
        [
            "rna_id",
            "degree",
            "betweenness",
            "pagerank"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("Top mRNA hubs:")

print(
    result[
        result["rna_type"] == "mRNA"
    ][
        [
            "rna_id",
            "degree",
            "betweenness",
            "pagerank"
        ]
    ]
    .head(10)
    .to_string(index=False)
)