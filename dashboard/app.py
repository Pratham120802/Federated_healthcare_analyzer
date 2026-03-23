import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import os
import streamlit as st
import matplotlib.pyplot as plt

from llm.llm_generator import generate_insight


st.title("Federated Healthcare Analyzer")

if not os.path.exists("metrics.json"):
    st.warning("No training metrics found yet. Run the federated server and clients first.")
    st.stop()

with open("metrics.json", "r") as f:
    metrics = json.load(f)

rounds = list(range(1, len(metrics) + 1))
accuracies = [m["accuracy"] for m in metrics]
latest_accuracy = accuracies[-1]
latest_loss = 1 - latest_accuracy  # placeholder if no real aggregated loss saved

st.metric("Latest Accuracy", f"{latest_accuracy:.4f}")
st.metric("Approx. Loss", f"{latest_loss:.4f}")

fig, ax = plt.subplots()
ax.plot(rounds, accuracies, marker="o")
ax.set_xlabel("Federated Rounds")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Across Rounds")
st.pyplot(fig)

st.subheader("LLM Clinical Insight")
if st.button("Generate Explanation"):
    with st.spinner("Generating clinical explanation..."):
     insight = generate_insight(latest_accuracy, latest_loss)
    st.write(insight)