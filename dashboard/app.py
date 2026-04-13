import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from federated.data_loader import (
    dataset_exists,
    delete_custom_dataset,
    get_dataset_summary,
    save_custom_dataset,
    set_data_config,
    read_uploaded_file,
    SUPPORTED_FORMATS,
)
from federated.run_training import run_federated_training
from federated.dataset_context import get_training_data_context
from llm.llm_generator import (
    generate_insight_with_history,
    ollama_chat,
    ollama_chat_stream,
    stream_initial_insight,
)

METRICS_PATH = _PROJECT_ROOT / "metrics.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Federated Healthcare Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Clean CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-card.orange { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
    .metric-card.blue { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        opacity: 0.9;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
from dashboard.auth import check_login, register_user, register_google_user

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")


def show_login_page():
    """Simple login page."""
    st.markdown("<h1 style='text-align: center; margin-top: 3rem;'>🏥 Federated Healthcare Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; margin-bottom: 2rem;'>Secure login to access the platform</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        tab1, tab2 = st.tabs(["Sign In", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In", use_container_width=True, type="primary"):
                    if username and password:
                        if check_login(username, password):
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                    else:
                        st.warning("Please enter username and password")
            
            st.caption("Demo: `admin` / `admin123`")
        
        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("Username", key="reg_user")
                new_email = st.text_input("Email", key="reg_email")
                new_pass = st.text_input("Password", type="password", key="reg_pass")
                new_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2")
                
                if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                    if not all([new_user, new_email, new_pass, new_pass2]):
                        st.warning("Please fill all fields")
                    elif new_pass != new_pass2:
                        st.error("Passwords don't match")
                    else:
                        success, msg = register_user(new_user, new_email, new_pass)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)


# Initialize session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    show_login_page()
    st.stop()

# Clear old metrics if no dataset exists (prevents showing stale data)
if not dataset_exists():
    if METRICS_PATH.exists():
        METRICS_PATH.unlink()
    # Also clear chat messages
    if "clinical_chat_messages" in st.session_state:
        st.session_state.clinical_chat_messages = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏥 FL Healthcare")
    st.caption(f"Logged in as: **{st.session_state.username}**")
    
    if st.button("Logout", use_container_width=True):
        # Clear all session state
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.clinical_chat_messages = None
        # Clear data files
        if METRICS_PATH.exists():
            METRICS_PATH.unlink()
        st.rerun()
    
    st.divider()
    
    # Upload section
    st.subheader("📁 Upload Dataset")
    st.caption(f"Supported: {', '.join(SUPPORTED_FORMATS.keys()).upper()}")
    
    uploaded_file = st.file_uploader(
        "Choose file",
        type=list(SUPPORTED_FORMATS.keys()),
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            df = read_uploaded_file(uploaded_file)
            st.success(f"✓ {len(df):,} rows × {len(df.columns)} cols")
            
            with st.expander("Preview"):
                st.dataframe(df.head(3), use_container_width=True)
            
            # Target selection
            target_col = st.selectbox(
                "Target column (to predict)",
                df.columns.tolist(),
                index=len(df.columns) - 1
            )
            
            n_classes = df[target_col].nunique()
            n_nulls = df[target_col].isna().sum()
            st.caption(f"{n_classes} classes" + (f", {n_nulls} nulls" if n_nulls > 0 else ""))
            
            if n_classes > 100:
                st.warning("Too many classes. Select a categorical column.")
            elif st.button("🚀 Start Training", type="primary", use_container_width=True):
                save_custom_dataset(df, uploaded_file.name)
                set_data_config(target_column=target_col)
                if METRICS_PATH.exists():
                    METRICS_PATH.unlink()
                st.session_state.clinical_chat_messages = None
                st.rerun()
                
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Active dataset
    if dataset_exists():
        st.divider()
        try:
            summary = get_dataset_summary()
            st.subheader("📊 Active Dataset")
            st.write(f"**{summary['name']}**")
            st.caption(f"Target: `{summary['target_column']}`")
            st.caption(f"{summary['n_samples']:,} samples, {summary['n_features']} features, {summary['n_classes']} classes")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Remove", use_container_width=True):
                    delete_custom_dataset()
                    if METRICS_PATH.exists():
                        METRICS_PATH.unlink()
                    st.session_state.clinical_chat_messages = None
                    st.rerun()
            with col2:
                if st.button("🔄 Retrain", use_container_width=True):
                    if METRICS_PATH.exists():
                        METRICS_PATH.unlink()
                    st.session_state.clinical_chat_messages = None
                    st.rerun()
        except:
            pass
    
    st.divider()
    
    # Reset button
    if st.button("🔄 Start Fresh", use_container_width=True, help="Clear all data and start over"):
        delete_custom_dataset()
        if METRICS_PATH.exists():
            METRICS_PATH.unlink()
        st.session_state.clinical_chat_messages = None
        st.rerun()
    
    st.caption("Federated Healthcare Analyzer v2.0")


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.title("🏥 Federated Healthcare Analyzer")
st.caption("Train healthcare models using federated learning across distributed clients")

# Check if dataset exists
if not dataset_exists():
    st.markdown("---")
    
    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h2 style="color: #667eea;">Welcome! 👋</h2>
            <p style="color: #666; font-size: 1.1rem; margin: 1rem 0 2rem 0;">
                Upload a healthcare dataset to get started with federated learning analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("### 📋 How It Works")
    
    step1, step2, step3, step4 = st.columns(4)
    
    with step1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; height: 150px;">
            <div style="font-size: 2rem;">📁</div>
            <h4>1. Upload</h4>
            <p style="font-size: 0.85rem; color: #666;">Upload your dataset (CSV, Excel, JSON)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with step2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; height: 150px;">
            <div style="font-size: 2rem;">🎯</div>
            <h4>2. Select Target</h4>
            <p style="font-size: 0.85rem; color: #666;">Choose what you want to predict</p>
        </div>
        """, unsafe_allow_html=True)
    
    with step3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; height: 150px;">
            <div style="font-size: 2rem;">🔄</div>
            <h4>3. Train</h4>
            <p style="font-size: 0.85rem; color: #666;">Federated learning across 3 clients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with step4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; height: 150px;">
            <div style="font-size: 2rem;">📊</div>
            <h4>4. Analyze</h4>
            <p style="font-size: 0.85rem; color: #666;">View results and AI insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Supported formats
    st.markdown("### 📄 Supported Formats")
    st.markdown("CSV, XLSX, XLS, JSON, Parquet, TSV, TXT")
    
    st.markdown("---")
    st.info("👈 **Get started:** Upload a dataset using the sidebar")
    
    st.stop()

# Training
if not METRICS_PATH.exists():
    st.header("🔄 Training in Progress")
    
    try:
        summary = get_dataset_summary()
        st.info(f"Dataset: **{summary['n_samples']:,}** samples, **{summary['n_features']}** features, **{summary['n_classes']}** classes")
        
        progress = st.progress(0, "Initializing...")
        
        with st.spinner("Running federated training (20 rounds × 3 clients)..."):
            progress.progress(20, "Training...")
            metrics = run_federated_training(num_rounds=20, num_clients=3, local_epochs=10)
            progress.progress(100, "Complete!")
        
        st.success(f"✓ Training complete! Final accuracy: **{metrics[-1]['accuracy']*100:.2f}%**")
        st.rerun()
        
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

# Load metrics
with open(METRICS_PATH) as f:
    metrics = json.load(f)

if not metrics:
    st.warning("No metrics found")
    st.stop()

# Results
accuracies = [m["accuracy"] for m in metrics]
losses = [m.get("loss", 1 - m["accuracy"]) for m in metrics]
latest_acc = accuracies[-1]
latest_loss = losses[-1]
n_rounds = len(metrics)

# KPI Cards
st.header("📈 Performance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    delta = (accuracies[-1] - accuracies[0]) * 100 if len(accuracies) > 1 else 0
    st.markdown(f"""
    <div class="metric-card green">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">{latest_acc*100:.1f}%</div>
        <div class="metric-label">{"↑" if delta >= 0 else "↓"} {abs(delta):.1f}pp</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card orange">
        <div class="metric-label">Loss</div>
        <div class="metric-value">{latest_loss:.4f}</div>
        <div class="metric-label">Lower is better</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card blue">
        <div class="metric-label">Rounds</div>
        <div class="metric-value">{n_rounds}</div>
        <div class="metric-label">Completed</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    n_clients = metrics[-1].get("num_clients", 3)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Clients</div>
        <div class="metric-value">{n_clients}</div>
        <div class="metric-label">Hospitals</div>
    </div>
    """, unsafe_allow_html=True)

# Charts
st.header("📊 Training Progress")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=list(range(1, n_rounds + 1)),
        y=[a * 100 for a in accuracies],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#11998e', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(17, 153, 142, 0.1)'
    ))
    fig_acc.update_layout(
        title="Accuracy per Round",
        xaxis_title="Round",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig_acc, use_container_width=True)

with chart_col2:
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=list(range(1, n_rounds + 1)),
        y=losses,
        mode='lines+markers',
        name='Loss',
        line=dict(color='#f2994a', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(242, 153, 74, 0.1)'
    ))
    fig_loss.update_layout(
        title="Loss per Round",
        xaxis_title="Round",
        yaxis_title="Loss",
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig_loss, use_container_width=True)

# Round details
st.header("📋 Round Details")

round_df = pd.DataFrame([
    {
        "Round": m["round"],
        "Accuracy": f"{m['accuracy']*100:.2f}%",
        "Loss": f"{m.get('loss', 0):.4f}",
        "Clients": m.get("num_clients", 3),
        "Samples": f"{m.get('total_examples', 'N/A'):,}" if isinstance(m.get('total_examples'), int) else "N/A"
    }
    for m in metrics
])
st.dataframe(round_df, use_container_width=True, hide_index=True)

# AI Insights
st.header("🤖 AI Clinical Insights")

if "clinical_chat_messages" not in st.session_state:
    st.session_state.clinical_chat_messages = None

_llm_data_context = get_training_data_context()

if st.session_state.clinical_chat_messages is None:
    if st.button("✨ Generate AI Insight", type="primary"):
        with st.spinner("Generating insight..."):
            try:
                msgs, err = generate_insight_with_history(
                    accuracy=latest_acc,
                    loss=latest_loss,
                    num_rounds=n_rounds,
                    accuracy_history=accuracies,
                    data_context=_llm_data_context,
                )
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.session_state.clinical_chat_messages = msgs
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
else:
    # Show chat
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Regenerate"):
            st.session_state.clinical_chat_messages = None
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.clinical_chat_messages = None
            st.rerun()
    
    # Display messages
    for msg in st.session_state.clinical_chat_messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a follow-up question..."):
        st.session_state.clinical_chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""
            for chunk in ollama_chat_stream(st.session_state.clinical_chat_messages, max_tokens=200):
                response += chunk
                placeholder.markdown(response + "▌")
            placeholder.markdown(response)
        
        if response and "[Error:" not in response:
            st.session_state.clinical_chat_messages.append({"role": "assistant", "content": response})
        else:
            st.session_state.clinical_chat_messages.pop()

# Footer
st.divider()
st.caption("Federated Healthcare Analyzer — Powered by Flower, PyTorch & Streamlit")
