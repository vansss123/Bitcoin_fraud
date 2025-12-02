import pandas as pd
import numpy as np
import networkx as nx
import joblib
import plotly.graph_objects as go

# -----------------------------
# Paths inside the Space
# -----------------------------
FEATURES_PATH = r"C:\Users\vansh\Desktop\DSSA\Sem3_assign\Banking\m\data\elliptic_txs_features.csv"
CLASSES_PATH  = r"C:\Users\vansh\Desktop\DSSA\Sem3_assign\Banking\m\data\elliptic_txs_classes.csv"
EDGES_PATH    = r"C:\Users\vansh\Desktop\DSSA\Sem3_assign\Banking\m\data\elliptic_txs_edgelist.csv"
MODEL_PATH    = r"C:\Users\vansh\Desktop\DSSA\Sem3_assign\Banking\m\Models\lightgbm_final_model.pkl"

# -----------------------------
# Class label mapping
# -----------------------------
CLASS_LABELS = {
    1: "Illicit",
    2: "Licit",
    0: "Unknown",
    None: "Unknown"
}

# -----------------------------
# 1. Load & prepare data
# -----------------------------
features_raw = pd.read_csv(FEATURES_PATH, header=None)

num_cols  = features_raw.shape[1]
num_feats = num_cols - 2  # txId + time_step

feat_cols = [f"f_{i}" for i in range(1, num_feats + 1)]
rename_map = {0: "txId", 1: "time_step"}
rename_map.update({i + 2: feat_cols[i] for i in range(num_feats)})

features_df = features_raw.rename(columns=rename_map)

# Classes file: HAS header
classes_df = pd.read_csv(CLASSES_PATH)
classes_df = classes_df.rename(columns={
    classes_df.columns[0]: "txId",
    classes_df.columns[1]: "class"
})

# Edges file: HAS header (txId1, txId2)
edges_df = pd.read_csv(EDGES_PATH)
edges_df = edges_df.rename(columns={
    edges_df.columns[0]: "src",
    edges_df.columns[1]: "dst"
})

# Ensure consistent types
features_df["txId"] = features_df["txId"].astype(int)
classes_df["txId"]  = classes_df["txId"].astype(int)
edges_df["src"]     = edges_df["src"].astype(int)
edges_df["dst"]     = edges_df["dst"].astype(int)

# Merge features + classes
data_df = features_df.merge(classes_df, on="txId", how="left")

# -----------------------------
# 2. Load model & compute risk
# -----------------------------
model = joblib.load(MODEL_PATH)

X_full = data_df[feat_cols].values.astype("float32")
all_proba = model.predict_proba(X_full)[:, 1]
data_df["proba_illicit"] = all_proba

# Risk bucket function
def risk_bucket(p: float) -> str:
    if p >= 0.90:
        return "High"
    elif p >= 0.75:
        return "Medium"
    elif p >= 0.50:
        return "Low"
    else:
        return "Safe"

data_df["risk_bucket"] = data_df["proba_illicit"].apply(risk_bucket)

# -----------------------------
# 3. Build graph & degree
# -----------------------------
G_full = nx.Graph()
G_full.add_edges_from(edges_df[["src", "dst"]].itertuples(index=False, name=None))

degree_dict = dict(G_full.degree())
data_df["degree"] = data_df["txId"].map(degree_dict).fillna(0).astype(int)

# Mappings for fast lookup
proba_map = data_df.set_index("txId")["proba_illicit"].to_dict()
bucket_map = data_df.set_index("txId")["risk_bucket"].to_dict()

# UPDATED — convert numeric class to text label
class_raw = data_df.set_index("txId")["class"].to_dict()
class_map = {tx: CLASS_LABELS.get(v, "Unknown") for tx, v in class_raw.items()}

degree_map = data_df.set_index("txId")["degree"].to_dict()

# -----------------------------
# 4. Helper: predict a single tx
# -----------------------------
def predict_tx(txid: int) -> dict:
    txid = int(txid)

    if txid not in proba_map:
        raise ValueError(f"txId {txid} not found in dataset")

    p = float(proba_map[txid])
    bucket = bucket_map.get(txid, "Safe")
    degree = int(degree_map.get(txid, 0))

    # UPDATED — use textual class label
    cls = class_map.get(txid, "Unknown")

    return {
        "txId": txid,
        "proba_illicit": p,
        "risk_bucket": bucket,
        "degree": degree,
        "class": cls,  # <-- NOW RETURNS LABEL
    }

# -----------------------------
# 5. Neighbor risk distribution
# -----------------------------
def neighbor_risk_distribution(txid: int):
    txid = int(txid)
    if txid not in G_full:
        raise ValueError(f"txId {txid} not found in graph")

    neighbors = list(G_full.neighbors(txid))
    if not neighbors:
        return {}

    nb_df = data_df.set_index("txId").loc[neighbors]
    counts = nb_df["risk_bucket"].value_counts().to_dict()
    return counts

# -----------------------------
# 6. Overall risk distribution
# -----------------------------
def overall_risk_distribution():
    return data_df["risk_bucket"].value_counts().to_dict()

# -----------------------------
# 7. 3D ego graph with Plotly
# -----------------------------
def plot_3d_tx_ego(txid: int, hops: int = 2, max_nodes: int = 400):
    txid = int(txid)

    if txid not in G_full:
        raise ValueError(f"txId {txid} not found in graph")

    ego = nx.ego_graph(G_full, txid, radius=hops)

    if ego.number_of_nodes() > max_nodes:
        nodes = list(ego.nodes())
        nodes.remove(txid)
        nodes_sorted = sorted(nodes, key=lambda n: ego.degree[n], reverse=True)
        keep = [txid] + nodes_sorted[:max_nodes - 1]
        ego = ego.subgraph(keep).copy()

    pos = nx.spring_layout(ego, dim=3, seed=42)

    node_list = list(ego.nodes())
    xs = [pos[n][0] for n in node_list]
    ys = [pos[n][1] for n in node_list]
    zs = [pos[n][2] for n in node_list]

    probs = [proba_map.get(n, 0.0) for n in node_list]

    # Edges
    edge_x, edge_y, edge_z = [], [], []
    for u, v in ego.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=1, color="lightgray"),
        hoverinfo="none",
        showlegend=False,
    )

    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(
            size=6,
            color=probs,
            colorscale="RdYlGn_r",
            cmin=0.0,
            cmax=1.0,
            colorbar=dict(title="p(illicit)"),
            opacity=0.9,
        ),
        text=[
            f"txId: {n}<br>p_illicit: {proba_map.get(n, 0.0):.4f}"
            for n in node_list
        ],
        hoverinfo="text",
        showlegend=False,
    )

    sx, sy, sz = pos[txid]
    seed_trace = go.Scatter3d(
        x=[sx],
        y=[sy],
        z=[sz],
        mode="markers",
        marker=dict(
            size=10,
            color="#1f77b4",
            line=dict(color="black", width=2),
        ),
        text=[
            f"SEED txId: {txid}<br>p_illicit: {proba_map.get(txid, 0.0):.4f}"
        ],
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace, seed_trace])
    fig.update_layout(
        title=f"3D Transaction Ego Graph (Seed {txid}, hops={hops})",
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
