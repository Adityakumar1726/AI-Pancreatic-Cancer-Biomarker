import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

NODE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "interactions",
    "nodes.csv"
)

EDGE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "interactions",
    "edges.csv"
)

EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rgcn_node_embeddings.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

FINAL_FILE = os.path.join(
    OUTPUT_DIR,
    "dqn_driver_candidates.csv"
)

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "dqn_model.pt"
)


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 60)
print("DQN PANCREATIC CANCER DRIVER RNA SELECTION")
print("=" * 60)


# ============================================================
# 4. LOAD DATA
# ============================================================

nodes = pd.read_csv(
    NODE_FILE
)

edges = pd.read_csv(
    EDGE_FILE
)

embeddings = pd.read_csv(
    EMBEDDING_FILE
)

print("\nNodes:", len(nodes))
print("Edges:", len(edges))
print("Embeddings:", embeddings.shape)


# ============================================================
# 5. RGCN EMBEDDING COLUMNS
# ============================================================

embedding_columns = [
    c
    for c in embeddings.columns
    if c.startswith("rgcn_embedding_")
]

if len(embedding_columns) != 32:

    raise ValueError(
        f"Expected 32 RGCN embeddings, "
        f"found {len(embedding_columns)}"
    )


# ============================================================
# 6. ALIGN EMBEDDINGS
# ============================================================

merged = embeddings[
    [
        "node_id",
        "rna_id",
        "rna_type"
    ]
    + embedding_columns
].copy()

print(
    "\nAligned graph nodes:",
    len(merged)
)

print(
    "Embedding dimensions:",
    len(embedding_columns)
)


# ============================================================
# 7. CHECK DUPLICATE NODE IDs
# ============================================================

duplicate_count = (
    merged["node_id"]
    .duplicated()
    .sum()
)

if duplicate_count > 0:

    raise ValueError(
        f"Found {duplicate_count} duplicate node IDs "
        "in the RGCN embedding file."
    )


# ============================================================
# 8. NODE INDEX
# ============================================================

node_to_idx = {
    node_id: index
    for index, node_id
    in enumerate(
        merged["node_id"]
    )
}


# ============================================================
# 9. RGCN FEATURES
# ============================================================

X_full = torch.tensor(
    merged[
        embedding_columns
    ].values.astype(
        np.float32
    )
)

num_full_nodes = len(
    merged
)

embedding_dim = len(
    embedding_columns
)


# ============================================================
# 10. BUILD FULL ADJACENCY MATRIX
# ============================================================

adjacency_full = np.zeros(
    (
        num_full_nodes,
        num_full_nodes
    ),
    dtype=np.float32
)


for _, row in edges.iterrows():

    source = row["source"]
    target = row["target"]

    if (
        source in node_to_idx
        and target in node_to_idx
    ):

        i = node_to_idx[
            source
        ]

        j = node_to_idx[
            target
        ]

        weight = abs(
            float(
                row["weight"]
            )
        )

        adjacency_full[
            i,
            j
        ] = max(
            adjacency_full[
                i,
                j
            ],
            weight
        )

        adjacency_full[
            j,
            i
        ] = max(
            adjacency_full[
                j,
                i
            ],
            weight
        )


# ============================================================
# 11. FULL NETWORK DEGREE
# ============================================================

full_degree = adjacency_full.sum(
    axis=1
)


# ============================================================
# 12. REMOVE ISOLATED NODES
# ============================================================

connected_indices = np.where(
    full_degree > 0
)[0]


isolated_indices = np.where(
    full_degree <= 0
)[0]


print(
    "\nConnected candidate nodes:",
    len(connected_indices)
)

print(
    "Excluded isolated nodes:",
    len(isolated_indices)
)


if len(connected_indices) == 0:

    raise RuntimeError(
        "No connected nodes were found."
    )


# ============================================================
# 13. CREATE COMPACT CONNECTED-NODE NETWORK
# ============================================================

# This is the critical fix.
#
# The DQN will NEVER see isolated nodes.
#
# Full network:
#     1065 nodes
#
# DQN network:
#     521 connected nodes
# ============================================================

X = X_full[
    connected_indices
].clone()


adjacency = adjacency_full[
    np.ix_(
        connected_indices,
        connected_indices
    )
]


degree = full_degree[
    connected_indices
]


num_nodes = len(
    connected_indices
)


# ============================================================
# 14. NORMALIZED DEGREE
# ============================================================

degree_norm = (
    degree
    /
    (
        degree.max()
        + 1e-8
    )
)


# ============================================================
# 15. VERIFY COMPACT NETWORK
# ============================================================

if np.any(
    degree <= 0
):

    raise RuntimeError(
        "ERROR: Compact network still contains "
        "zero-degree nodes."
    )


print(
    "DQN action space:",
    num_nodes
)

print(
    "Minimum DQN node degree:",
    int(degree.min())
)

print(
    "Maximum DQN node degree:",
    int(degree.max())
)


# ============================================================
# 16. NETWORKX GRAPH
# ============================================================

graph = nx.from_numpy_array(
    adjacency
)


# ============================================================
# 17. PBH-INSPIRED SCORE
# ============================================================

def pbh_score(
    selected_nodes
):

    if len(
        selected_nodes
    ) == 0:

        return 0.0


    reachable = set()


    for driver in selected_nodes:

        reachable.add(
            driver
        )

        try:

            reachable.update(
                nx.single_source_shortest_path_length(
                    graph,
                    driver
                ).keys()
            )

        except Exception:

            pass


    coverage = (
        len(reachable)
        /
        num_nodes
    )


    efficiency = (
        coverage
        /
        np.sqrt(
            len(selected_nodes)
        )
    )


    return float(
        min(
            1.0,
            efficiency
        )
    )


# ============================================================
# 18. DQN MODEL
# ============================================================

class DQN(
    nn.Module
):

    def __init__(
        self,
        state_dim,
        action_dim
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                state_dim,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                action_dim
            )
        )


    def forward(
        self,
        state
    ):

        return self.network(
            state
        )


# ============================================================
# 19. STATE REPRESENTATION
# ============================================================

def create_state(
    selected_nodes
):

    if len(
        selected_nodes
    ) == 0:

        mean_embedding = torch.zeros(
            embedding_dim,
            dtype=torch.float32
        )

    else:

        mean_embedding = X[
            selected_nodes
        ].mean(
            dim=0
        )


    coverage = (
        len(selected_nodes)
        /
        MAX_DRIVERS
    )


    return torch.cat(
        [
            mean_embedding,

            torch.tensor(
                [
                    coverage
                ],
                dtype=torch.float32
            )
        ]
    )


# ============================================================
# 20. DQN PARAMETERS
# ============================================================

STATE_DIM = (
    embedding_dim
    + 1
)

ACTION_DIM = num_nodes


policy_net = DQN(
    STATE_DIM,
    ACTION_DIM
)


target_net = DQN(
    STATE_DIM,
    ACTION_DIM
)


target_net.load_state_dict(
    policy_net.state_dict()
)

target_net.eval()


optimizer = torch.optim.Adam(
    policy_net.parameters(),
    lr=0.001
)


# ============================================================
# 21. TRAINING PARAMETERS
# ============================================================

replay_buffer = []

MAX_BUFFER = 5000

BATCH_SIZE = 32

EPISODES = 100

MAX_DRIVERS = 10

GAMMA = 0.95

EPSILON = 1.0

EPSILON_MIN = 0.05

EPSILON_DECAY = 0.97


# ============================================================
# 22. ACTION SELECTION
# ============================================================

def choose_action(
    state,
    available_actions,
    epsilon
):

    if len(
        available_actions
    ) == 0:

        raise RuntimeError(
            "No available connected nodes."
        )


    # --------------------------------------------------------
    # Exploration
    # --------------------------------------------------------

    if random.random() < epsilon:

        return random.choice(
            available_actions
        )


    # --------------------------------------------------------
    # Exploitation
    # --------------------------------------------------------

    with torch.no_grad():

        q_values = policy_net(
            state.unsqueeze(0)
        )[0]


    # Only available connected nodes are considered

    candidate_q_values = q_values[
        available_actions
    ]


    best_position = torch.argmax(
        candidate_q_values
    ).item()


    return int(
        available_actions[
            best_position
        ]
    )


# ============================================================
# 23. TRAIN DQN
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING DQN")
print("=" * 60)


for episode in range(
    1,
    EPISODES + 1
):


    selected = []

    total_reward = 0.0

    old_score = 0.0


    # --------------------------------------------------------
    # Episode steps
    # --------------------------------------------------------

    for step in range(
        MAX_DRIVERS
    ):


        state = create_state(
            selected
        )


        # ----------------------------------------------------
        # AVAILABLE ACTIONS
        # ----------------------------------------------------

        available = [
            i
            for i in range(
                num_nodes
            )
            if i not in selected
        ]


        if len(
            available
        ) == 0:

            break


        # ----------------------------------------------------
        # ACTION
        # ----------------------------------------------------

        action = choose_action(
            state,
            available,
            EPSILON
        )


        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if action < 0 or action >= num_nodes:

            raise RuntimeError(
                "Invalid DQN action."
            )


        if degree[action] <= 0:

            raise RuntimeError(
                "DQN selected a zero-degree node."
            )


        # ----------------------------------------------------
        # NEW STATE
        # ----------------------------------------------------

        new_selected = (
            selected
            +
            [action]
        )


        # ----------------------------------------------------
        # PBH-INSPIRED SCORE
        # ----------------------------------------------------

        new_score = pbh_score(
            new_selected
        )


        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        reward = (
            new_score
            -
            old_score
        )


        reward += (
            0.01
            *
            degree_norm[action]
        )


        # ----------------------------------------------------
        # NEXT STATE
        # ----------------------------------------------------

        next_state = create_state(
            new_selected
        )


        done = (
            step
            ==
            MAX_DRIVERS - 1
        )


        # ----------------------------------------------------
        # STORE EXPERIENCE
        # ----------------------------------------------------

        replay_buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


        if len(
            replay_buffer
        ) > MAX_BUFFER:

            replay_buffer.pop(
                0
            )


        selected = new_selected

        old_score = new_score

        total_reward += reward


        # ----------------------------------------------------
        # TRAIN FROM REPLAY BUFFER
        # ----------------------------------------------------

        if len(
            replay_buffer
        ) >= BATCH_SIZE:


            batch = random.sample(
                replay_buffer,
                BATCH_SIZE
            )


            states = torch.stack(
                [
                    item[0]
                    for item in batch
                ]
            )


            actions = torch.tensor(
                [
                    item[1]
                    for item in batch
                ],
                dtype=torch.long
            )


            rewards = torch.tensor(
                [
                    item[2]
                    for item in batch
                ],
                dtype=torch.float32
            )


            next_states = torch.stack(
                [
                    item[3]
                    for item in batch
                ]
            )


            dones = torch.tensor(
                [
                    item[4]
                    for item in batch
                ],
                dtype=torch.float32
            )


            # ------------------------------------------------
            # CURRENT Q
            # ------------------------------------------------

            current_q = (
                policy_net(
                    states
                )
                .gather(
                    1,
                    actions.unsqueeze(1)
                )
                .squeeze(1)
            )


            # ------------------------------------------------
            # NEXT Q
            # ------------------------------------------------

            with torch.no_grad():

                next_q_values = target_net(
                    next_states
                )


                next_q = (
                    next_q_values
                    .max(
                        dim=1
                    )[0]
                )


                target_q = (
                    rewards
                    +
                    GAMMA
                    *
                    next_q
                    *
                    (
                        1
                        -
                        dones
                    )
                )


            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            loss = F.mse_loss(
                current_q,
                target_q
            )


            optimizer.zero_grad()

            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                policy_net.parameters(),
                1.0
            )


            optimizer.step()


    # --------------------------------------------------------
    # EPSILON DECAY
    # --------------------------------------------------------

    EPSILON = max(
        EPSILON_MIN,
        EPSILON
        *
        EPSILON_DECAY
    )


    # --------------------------------------------------------
    # TARGET NETWORK
    # --------------------------------------------------------

    if episode % 10 == 0:

        target_net.load_state_dict(
            policy_net.state_dict()
        )


        print(
            f"Episode {episode:03d}/{EPISODES} "
            f"| Reward: {total_reward:.6f} "
            f"| PBH Score: {old_score:.6f} "
            f"| Epsilon: {EPSILON:.4f}"
        )


# ============================================================
# 24. FINAL DQN DRIVER SELECTION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL DQN DRIVER SELECTION")
print("=" * 60)


policy_net.eval()


selected = []

selected_q_values = []


state = create_state(
    selected
)


for rank in range(
    MAX_DRIVERS
):


    # --------------------------------------------------------
    # ONLY CONNECTED NODES EXIST IN THIS ACTION SPACE
    # --------------------------------------------------------

    available = [
        i
        for i in range(
            num_nodes
        )
        if i not in selected
    ]


    if len(
        available
    ) == 0:

        break


    # --------------------------------------------------------
    # Q VALUES
    # --------------------------------------------------------

    with torch.no_grad():

        q_values = policy_net(
            state.unsqueeze(0)
        )[0]


    # --------------------------------------------------------
    # SELECT BEST AVAILABLE ACTION
    # --------------------------------------------------------

    candidate_q_values = q_values[
        available
    ]


    best_position = torch.argmax(
        candidate_q_values
    ).item()


    action = available[
        best_position
    ]


    q_value = float(
        q_values[
            action
        ].item()
    )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if action < 0:

        raise RuntimeError(
            "Invalid negative action."
        )


    if action >= num_nodes:

        raise RuntimeError(
            "Invalid action outside DQN network."
        )


    if degree[action] <= 0:

        raise RuntimeError(
            "CRITICAL ERROR: "
            "Zero-degree node selected."
        )


    if action in selected:

        raise RuntimeError(
            "Duplicate node selected."
        )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    selected.append(
        action
    )

    selected_q_values.append(
        q_value
    )


    # --------------------------------------------------------
    # UPDATE STATE
    # --------------------------------------------------------

    state = create_state(
        selected
    )


# ============================================================
# 25. FINAL PBH-INSPIRED SCORE
# ============================================================

final_pbh_score = pbh_score(
    selected
)


# ============================================================
# 26. FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("SELECTED NODE VALIDATION")
print("=" * 60)


for rank, node_index in enumerate(
    selected,
    start=1
):


    original_index = (
        connected_indices[
            node_index
        ]
    )


    row = merged.iloc[
        original_index
    ]


    node_degree = int(
        degree[
            node_index
        ]
    )


    print(
        f"{rank:02d}. "
        f"{row['rna_id']} "
        f"| {row['rna_type']} "
        f"| degree = {node_degree}"
    )


    if node_degree <= 0:

        raise RuntimeError(
            "FINAL VALIDATION FAILED."
        )


print("\n")
print(
    "Validation passed:"
)

print(
    "All selected DQN drivers have network_degree > 0."
)


# ============================================================
# 27. CREATE RESULTS
# ============================================================

results = []


for rank, node_index in enumerate(
    selected,
    start=1
):


    # Convert compact DQN index
    # back to original graph index

    original_index = (
        connected_indices[
            node_index
        ]
    )


    row = merged.iloc[
        original_index
    ]


    results.append(
        {
            "dqn_rank":
                rank,

            "node_id":
                row["node_id"],

            "rna_id":
                row["rna_id"],

            "rna_type":
                row["rna_type"],

            "network_degree":
                int(
                    degree[
                        node_index
                    ]
                ),

            "degree_score":
                float(
                    degree_norm[
                        node_index
                    ]
                ),

            "dqn_q_value":
                selected_q_values[
                    rank - 1
                ],

            "final_pbh_score":
                final_pbh_score
        }
    )


# ============================================================
# 28. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 29. FINAL DATAFRAME CHECK
# ============================================================

if len(
    results_df
) != MAX_DRIVERS:

    raise RuntimeError(
        f"Expected {MAX_DRIVERS} drivers, "
        f"but obtained {len(results_df)}."
    )


if (
    results_df[
        "network_degree"
    ]
    <= 0
).any():

    raise RuntimeError(
        "Final result contains a zero-degree node."
    )


# ============================================================
# 30. SAVE RESULTS
# ============================================================

results_df.to_csv(
    FINAL_FILE,
    index=False
)


# ============================================================
# 31. SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            policy_net.state_dict(),

        "state_dim":
            STATE_DIM,

        "action_dim":
            ACTION_DIM,

        "embedding_dim":
            embedding_dim,

        "selected_driver_nodes":
            selected,

        "connected_candidate_count":
            len(
                connected_indices
            ),

        "excluded_isolated_count":
            len(
                isolated_indices
            ),

        "final_pbh_score":
            final_pbh_score
    },
    MODEL_FILE
)


# ============================================================
# 32. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL DQN-SELECTED DRIVERS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


print("\n")
print("Final PBH-inspired score:")

print(
    final_pbh_score
)


print("\n")
print("Results saved:")

print(
    FINAL_FILE
)


print("\n")
print("Model saved:")

print(
    MODEL_FILE
)


# ============================================================
# 33. COMPLETION
# ============================================================

print("\n")
print("=" * 60)
print("DQN DRIVER-SELECTION STAGE COMPLETED")
print("=" * 60)