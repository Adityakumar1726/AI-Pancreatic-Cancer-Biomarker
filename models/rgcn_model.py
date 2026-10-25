import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NODE_FILE = os.path.join(BASE_DIR, "data", "interactions", "nodes.csv")
EDGE_FILE = os.path.join(BASE_DIR, "data", "interactions", "edges.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBEDDING_FILE = os.path.join(
    OUTPUT_DIR,
    "rgcn_node_embeddings.csv"
)

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "rgcn_model.pt"
)


# ============================================================
# 2. LOAD GRAPH
# ============================================================

print("=" * 60)
print("RGCN PANCREATIC CANCER BIOMARKER MODEL")
print("=" * 60)

nodes = pd.read_csv(NODE_FILE)
edges = pd.read_csv(EDGE_FILE)

print("\nNodes:", len(nodes))
print("Edges:", len(edges))

print("\nNode columns:")
print(list(nodes.columns))

print("\nRelation types:")
print(edges["relation"].value_counts())


# ============================================================
# 3. CREATE NODE INDEX
# ============================================================

node_to_idx = {
    node_id: i
    for i, node_id in enumerate(nodes["node_id"])
}

nodes["index"] = nodes["node_id"].map(node_to_idx)

# Check for missing nodes
if nodes["index"].isna().any():
    raise ValueError("Some nodes could not be indexed.")


# ============================================================
# 4. RELATION ENCODING
# ============================================================

relations = sorted(edges["relation"].unique())

relation_to_id = {
    relation: i
    for i, relation in enumerate(relations)
}

print("\nRelation mapping:")

for relation, relation_id in relation_to_id.items():
    print(relation, "->", relation_id)


# ============================================================
# 5. CREATE EDGE INDEX
# ============================================================

source_indices = edges["source"].map(node_to_idx)
target_indices = edges["target"].map(node_to_idx)

valid_edges = (
    source_indices.notna()
    & target_indices.notna()
)

edges = edges.loc[valid_edges].copy()

source_indices = source_indices.loc[valid_edges].astype(int)
target_indices = target_indices.loc[valid_edges].astype(int)

edge_index = torch.tensor(
    np.vstack([
        source_indices.values,
        target_indices.values
    ]),
    dtype=torch.long
)

edge_type = torch.tensor(
    edges["relation"].map(relation_to_id).values,
    dtype=torch.long
)

edge_weight = torch.tensor(
    edges["weight"].values,
    dtype=torch.float
)


# ============================================================
# 6. CREATE INITIAL NODE FEATURES
# ============================================================

# RNA type one-hot:
# mRNA   = [1,0,0]
# lncRNA = [0,1,0]
# miRNA  = [0,0,1]

rna_types = ["mRNA", "lncRNA", "miRNA"]

type_features = np.zeros(
    (len(nodes), len(rna_types)),
    dtype=np.float32
)

for i, rna_type in enumerate(nodes["rna_type"]):
    if rna_type in rna_types:
        type_features[i, rna_types.index(rna_type)] = 1.0


# ------------------------------------------------------------
# Calculate node degree
# ------------------------------------------------------------

degree = np.zeros(len(nodes), dtype=np.float32)

for src, dst in zip(
    source_indices.values,
    target_indices.values
):
    degree[src] += 1
    degree[dst] += 1


# Log-transform degree
degree = np.log1p(degree)

# Normalize
if degree.max() > 0:
    degree = degree / degree.max()

degree = degree.reshape(-1, 1)


# Combine features
x_numpy = np.concatenate(
    [
        type_features,
        degree
    ],
    axis=1
)

x = torch.tensor(
    x_numpy,
    dtype=torch.float
)

print("\nInitial feature shape:", x.shape)


# ============================================================
# 7. CREATE PYTORCH GEOMETRIC GRAPH
# ============================================================

data = Data(
    x=x,
    edge_index=edge_index,
    edge_type=edge_type
)

print("\nPyTorch Geometric graph created.")
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)


# ============================================================
# 8. RGCN MODEL
# ============================================================

class RGCNEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_relations
    ):
        super().__init__()

        self.conv1 = RGCNConv(
            input_dim,
            hidden_dim,
            num_relations
        )

        self.conv2 = RGCNConv(
            hidden_dim,
            embedding_dim,
            num_relations
        )

        self.dropout = 0.2

    def forward(
        self,
        x,
        edge_index,
        edge_type
    ):

        x = self.conv1(
            x,
            edge_index,
            edge_type
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index,
            edge_type
        )

        return x


# ============================================================
# 9. RELATION-AWARE DECODER
# ============================================================

class RelationDecoder(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_relations
    ):
        super().__init__()

        self.relation_embeddings = nn.Parameter(
            torch.randn(
                num_relations,
                embedding_dim
            )
        )

    def forward(
        self,
        z,
        src,
        dst,
        relation
    ):

        src_z = z[src]
        dst_z = z[dst]

        relation_z = self.relation_embeddings[
            relation
        ]

        score = (
            src_z
            * relation_z
            * dst_z
        ).sum(dim=1)

        return score


# ============================================================
# 10. BUILD MODEL
# ============================================================

INPUT_DIM = x.shape[1]
HIDDEN_DIM = 64
EMBEDDING_DIM = 32

NUM_RELATIONS = len(relations)

encoder = RGCNEncoder(
    INPUT_DIM,
    HIDDEN_DIM,
    EMBEDDING_DIM,
    NUM_RELATIONS
)

decoder = RelationDecoder(
    EMBEDDING_DIM,
    NUM_RELATIONS
)


# ============================================================
# 11. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    list(encoder.parameters())
    + list(decoder.parameters()),
    lr=0.01,
    weight_decay=5e-4
)


# ============================================================
# 12. NEGATIVE EDGE SAMPLING
# ============================================================

def generate_negative_edges(
    num_nodes,
    num_edges
):

    src = torch.randint(
        0,
        num_nodes,
        (num_edges,)
    )

    dst = torch.randint(
        0,
        num_nodes,
        (num_edges,)
    )

    relation = torch.randint(
        0,
        NUM_RELATIONS,
        (num_edges,)
    )

    return src, dst, relation


# ============================================================
# 13. TRAIN RGCN
# ============================================================

EPOCHS = 200

print("\n")
print("=" * 60)
print("TRAINING RGCN")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    encoder.train()
    decoder.train()

    optimizer.zero_grad()

    # --------------------------------------------------------
    # Generate node embeddings
    # --------------------------------------------------------

    z = encoder(
        data.x,
        data.edge_index,
        data.edge_type
    )

    # --------------------------------------------------------
    # Positive edges
    # --------------------------------------------------------

    positive_src = data.edge_index[0]
    positive_dst = data.edge_index[1]
    positive_relation = data.edge_type

    positive_score = decoder(
        z,
        positive_src,
        positive_dst,
        positive_relation
    )

    # --------------------------------------------------------
    # Negative edges
    # --------------------------------------------------------

    negative_src, negative_dst, negative_relation = \
        generate_negative_edges(
            data.num_nodes,
            data.num_edges
        )

    negative_score = decoder(
        z,
        negative_src,
        negative_dst,
        negative_relation
    )

    # --------------------------------------------------------
    # Binary classification loss
    # --------------------------------------------------------

    positive_labels = torch.ones_like(
        positive_score
    )

    negative_labels = torch.zeros_like(
        negative_score
    )

    loss_positive = F.binary_cross_entropy_with_logits(
        positive_score,
        positive_labels
    )

    loss_negative = F.binary_cross_entropy_with_logits(
        negative_score,
        negative_labels
    )

    loss = (
        loss_positive
        + loss_negative
    ) / 2

    loss.backward()

    optimizer.step()

    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if epoch == 1 or epoch % 20 == 0:

        print(
            f"Epoch {epoch:03d}/{EPOCHS} "
            f"| Loss: {loss.item():.6f}"
        )


# ============================================================
# 14. GENERATE FINAL RGCN EMBEDDINGS
# ============================================================

encoder.eval()

with torch.no_grad():

    embeddings = encoder(
        data.x,
        data.edge_index,
        data.edge_type
    )


# ============================================================
# 15. SAVE EMBEDDINGS
# ============================================================

embedding_array = embeddings.numpy()

embedding_columns = [
    f"rgcn_embedding_{i+1}"
    for i in range(EMBEDDING_DIM)
]

embedding_df = pd.DataFrame(
    embedding_array,
    columns=embedding_columns
)

embedding_df.insert(
    0,
    "rna_type",
    nodes["rna_type"].values
)

embedding_df.insert(
    0,
    "rna_id",
    nodes["rna_id"].values
)

embedding_df.insert(
    0,
    "node_id",
    nodes["node_id"].values
)

embedding_df.to_csv(
    EMBEDDING_FILE,
    index=False
)


# ============================================================
# 16. SAVE MODEL
# ============================================================

torch.save(
    {
        "encoder_state_dict":
            encoder.state_dict(),

        "decoder_state_dict":
            decoder.state_dict(),

        "relations":
            relation_to_id,

        "input_dim":
            INPUT_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "embedding_dim":
            EMBEDDING_DIM
    },
    MODEL_FILE
)


# ============================================================
# 17. FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("RGCN TRAINING COMPLETE")
print("=" * 60)

print("\nEmbedding shape:")
print(embeddings.shape)

print("\nEmbeddings saved:")
print(EMBEDDING_FILE)

print("\nModel saved:")
print(MODEL_FILE)

print("\nNumber of relations:")
print(NUM_RELATIONS)

print("\nRelation types:")
for relation in relations:
    print("-", relation)

print("\nRGCN stage completed successfully.")