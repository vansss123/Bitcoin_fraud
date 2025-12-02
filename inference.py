import pandas as pd
import numpy as np
import networkx as nx
import joblib
import plotly.graph_objects as go
from functools import lru_cache

# =============================
# Google Drive file IDs
# =============================

FEATURES_ID = "1Vyz05FjbzK6AzJoEPQjYyZq0n-dDIZqJ"   # elliptic_txs_features.csv
CLASSES_ID  = "1v0wJWmOpdggj1HXVtytyoOjvOtZIQF5u"   # elliptic_txs_classes.csv
EDGES_ID    = "1kxfTDSXmHh9EbJtCamNugaJ-pp4fx5Mj"   # elliptic_txs_edgelist.csv

# Model is stored inside the repo (Models folder in GitHub)
MODEL_PATH = "Models/lightgbm_final_model.pkl"

# Class label mapping
CLASS_LABELS = {
    1: "Illicit",
    2: "Licit",
    0: "Unknown",
    None: "Unknown",
}


def drive_url(file_id: str) -> str:
    """Build direct-download URL from a Google Drive file_id."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# =============================
# 1. Load & prepare data
# =============================
@lru_cache(maxsize=1)
def load_raw_data():
    """Load features, classes, edges from Google Drive and prepare them."""

    # FEATURES: this CSV now HAS a header row (txId, time_step, f_1, f_2, ...)
    features_raw = pd.read_csv(drive_url(FEATURES_ID), low_memory=False)

    # First two columns are txId and time_step, rest are features
    cols = list(features_raw.columns)

    if len(cols) < 3:
        raise ValueError("Features file does not have the expected columns.")

    rename_map = {
        cols[0]: "txId",
        cols[1]: "time_step",
    }

    feat_cols = []
    for i, c in enumerate(cols[2:], start=1):
        new_name = f"f_{i}"
        rename_map[c] = new_name
        feat_cols.append(new_name)

    features_df = features_raw.rename(columns=rename_map)

    # CLASSES: has header
    classes_df = pd.read_csv(drive_url(CLASSES_ID))
    classes_df = classes_df.rename(
        columns={
            classes_df.columns[0]: "txId",
            classes_df.columns[1]: "class",
        }
    )

    # EDGES: has header (txId1, txId2)
    edges_df = pd.read_csv(drive_url(EDGES_ID))
    edges_df = edges_df.rename(
        columns={
            edges_df.columns[0]: "src",
            edges_df.columns[1]: "dst",
        }
    )

    # Ensure consistent types
    features_df["txId"] = features_df["txId"].astype(int)
    classes_df["txId"] = classes_df["txId"].astype(int)
    edges_df["src"] = edges_df["src"].astype(int)
    edges_df["dst"] = edges_df["dst"].astype(int)

    # Merge features + classes
    data_df = features_df.merge(classes_df, on="txId", how="left")

    return data_df, edges_df, feat_cols


@lru_cache(maxsize=1)
def load_everything():
    """
    Load data, model, compute probabilities, risk buckets,
    graph, and all lookup dicts.
    """
    data_df, edges_df, feat_cols = load_raw_data()

    # Load model from repo
    model = joblib.load(MODEL_PATH)

    # --- CLEAN & ALIGN FEATURES BEFORE SENDING TO MODEL ---

    # Keep only feature columns and force numeric
    feats_df = data_df[feat_cols].apply(pd.to_numeric, errors="coerce")

    n_data = feats_df.shape[1]
    n_model = getattr(model, "n_features_in_", None)

    # Align number of features with what the model expects
    if n_model is not None:
        if n_data > n_model:
            # Too many features in the CSV -> keep only the first n_model
            feats_df = feats_df.iloc[:, :n_model]
        elif n_data < n_model:
            # Not enough features -> this is a real problem
            raise ValueError(
                f"Model expects {n_model} features but data only has {n_data}."
            )

    # Replace NaNs with 0 and convert to float32
    X_full = feats_df.fillna(0).astype("float32").values

    # Predict probabilities for all transactions
    all_proba = model.predict_proba(X_full)[:, 1]
    data_df["proba_illicit"] = all_proba

    # Risk bucket
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

    # Build graph
    G_full = nx.Graph()
    G_full.add_edges_from(
        edges_df[["src", "dst"]].itertuples(index=False, name=None)
    )

    # Degree
    degree_dict = dict(G_full.degree())
    data_df["degree"] = (
        data_df["txId"].map(degree_dict).fillna(0).astype(int)
    )

    # Lookup maps
    proba_map = data_df.set_index("txId")["proba_illicit"].to_dict()
    bucket_map = data_df.set_index("txId")["risk_bucket"].to_dict()

    class_raw = data_df.set_index("txId")["class"].to_dict()
    class_map = {
        tx: CLASS_LABELS.get(v, "Unknown") for tx, v in class_raw.items()
    }

    degree_map = data_df.set_index("txId")["degree"].to_dict()

    return (
        data_df,
        G_full,
        proba_map,
        bucket_map,
        class_map,
        degree_map,
    )

# =============================
# 2. Public functions used by app.py
# =============================

def predict_tx(txid: int) -> dict:
    (
        data_df,
        G_full,
        proba_map,
        bucket_map,
        class_map,
        degree_map,
    ) = load_everything()

    txid = int(txid)

    if txid not in proba_map:
        raise ValueError(f"txId {txid} not found in dataset")

    p = float(proba_map[txid])
    bucket = bucket_map.get(txid, "Safe")
    degree = int(degree_map.get(txid, 0))
    cls = class_map.get(txid, "Unknown")

    return {
        "txId": txid,
        "proba_illicit": p,
        "risk_bucket": bucket,
        "degree": degree,
        "class": cls,
    }


def neighbor_risk_distribution(txid: int):
    (
        data_df,
        G_full,
        proba_map,
        bucket_map,
        class_map,
        degree_map,
    ) = load_everything()

    txid = int(txid)
    if txid not in G_full:
        raise ValueError(f"txId {txid} not found in graph")

    neighbors = list(G_full.neighbors(txid))
    if not neighbors:
        return {}

    nb_df = data_df.set_index("txId").loc[neighbors]
    counts = nb_df["risk_bucket"].value_counts().to_dict()
    return counts


def overall_risk_distribution():
    (data_df, *_rest) = load_everything()
    return data_df["risk_bucket"].value_counts().to_dict()


def plot_3d_tx_ego(txid: int, hops: int = 2, max_nodes: int = 400):
    (
        data_df,
        G_full,
        proba_map,
        bucket_map,
        class_map,
        degree_map,
    ) = load_everything()

    txid = int(txid)

    if txid not in G_full:
        raise ValueError(f"txId {txid} not found in graph")

    ego = nx.ego_graph(G_full, txid, radius=hops)

    if ego.number_of_nodes() > max_nodes:
        nodes = list(ego.nodes())
        nodes.remove(txid)
        nodes_sorted = sorted(nodes, key=lambda n: ego.degree[n], reverse=True)
        keep = [txid] + nodes_sorted[: max_nodes - 1]
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



