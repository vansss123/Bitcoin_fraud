import streamlit as st
from inference import (
    predict_tx,
    plot_3d_tx_ego,
    neighbor_risk_distribution,
    overall_risk_distribution,
)

st.set_page_config(page_title="Elliptic Bitcoin Fraud Explorer", layout="wide")

st.title("Bitcoin Transaction Risk Explorer")

st.markdown(
    """
    This app uses a trained **LightGBM model** on the Elliptic Bitcoin dataset to:
    - Predict **illicit probability** for a given transaction (txId)
    - Assign a **risk bucket** (Safe / Low / Medium / High)
    - Show the transaction's **degree** (number of neighbors)
    - Visualize a **3D ego graph** colored by risk
    """
)

# Sidebar: global stats
st.sidebar.header("Global Risk Distribution")
overall_counts = overall_risk_distribution()
if overall_counts:
    total = sum(overall_counts.values())
    for k, v in overall_counts.items():
        st.sidebar.write(f"{k}: {v} ({v/total*100:.1f}%)")

    st.sidebar.bar_chart(overall_counts)
else:
    st.sidebar.write("No risk data available.")

st.subheader("Analyze a Transaction")

tx_input = st.text_input("Enter transaction ID (txId)", value="78144215")
col_h1, col_h2 = st.columns(2)
with col_h1:
    hops = st.slider("Ego-graph hops", min_value=1, max_value=4, value=2)
with col_h2:
    max_nodes = st.slider("Max nodes in ego graph", min_value=50, max_value=800, value=400, step=50)

if st.button("Analyze"):
    try:
        txid = int(tx_input)

        info = predict_tx(txid)

        st.markdown("### Prediction Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("txId", str(info["txId"]))
        c2.metric("p(illicit)", f"{info['proba_illicit']:.6f}")
        c3.metric("Risk bucket", info["risk_bucket"])
        c4.metric("Degree", str(info["degree"]))
        c5.metric("Original class", str(info["class"]))

        st.markdown("### Neighbor Risk Distribution")
        nb_counts = neighbor_risk_distribution(txid)
        if nb_counts:
            st.bar_chart(nb_counts)
        else:
            st.write("This node has no neighbors in the graph.")

        st.markdown("### 3D Ego Graph")
        fig = plot_3d_tx_ego(txid, hops=hops, max_nodes=max_nodes)
        st.plotly_chart(fig, use_container_width=True)

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")
