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
from federated.dataset_context import get_training_data_context, get_dataset_info_for_display
from llm.llm_generator import (
    generate_insight_with_history,
    generate_multimodel_insight_with_history,
    ollama_chat,
    ollama_chat_stream,
    stream_initial_insight,
)

METRICS_PATH = _PROJECT_ROOT / "metrics.json"
MODEL_COMPARISON_PATH = _PROJECT_ROOT / "model_comparison.json"

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
from dashboard.auth import (
    check_login_local,
    check_login_supabase,
    register_user_local,
    register_user_supabase,
    reset_password,
    sign_out,
    get_auth_mode,
)


def _fin():
    _ = (
            "Google login session expired or is invalid. Please use “Continue with Google” again."
        )


def show_login_page():
    """Login page with Supabase and local auth support."""
    st.markdown("<h1 style='text-align: center; margin-top: 3rem;'>🏥 Federated Healthcare Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; margin-bottom: 2rem;'>Secure login to access the platform</p>", unsafe_allow_html=True)
    
    supabase_available = get_auth_mode() == "supabase"
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        # Let user choose auth method if Supabase is available
        if supabase_available:
            auth_choice = st.radio(
                "Authentication Method",
                ["Local", "Supabase"],
                horizontal=True,
                label_visibility="collapsed"
            )
            auth_mode = "supabase" if auth_choice == "Supabase" else "local"
        else:
            auth_mode = "local"
        
        # Show auth mode indicator
        if auth_mode == "supabase":
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <span style="background: linear-gradient(135deg, #3ECF8E 0%, #2B9E6F 100%); 
                            color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">
                    Powered by Supabase
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <span style="background: #666; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">
                    Local Authentication
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Sign In", "Register", "Reset Password"])

        with tab1:
            with st.form("login_form"):
                if auth_mode == "supabase":
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    
                    if st.form_submit_button("Sign In", use_container_width=True, type="primary"):
                        if email and password:
                            success, msg, user_data = check_login_supabase(email, password)
                            if success:
                                st.session_state.logged_in = True
                                st.session_state.username = email.split("@")[0]
                                st.session_state.user_email = email
                                st.session_state.user_id = user_data.get("id") if user_data else None
                                st.session_state.auth_mode = "supabase"
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.warning("Please enter email and password")
                else:
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    
                    if st.form_submit_button("Sign In", use_container_width=True, type="primary"):
                        if username and password:
                            if check_login_local(username, password):
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                st.session_state.auth_mode = "local"
                                st.rerun()
                            else:
                                st.error("Invalid credentials")
                        else:
                            st.warning("Please enter username and password")
            
            if auth_mode == "local":
                st.caption("Demo: `admin` / `admin123`")
        
        with tab2:
            with st.form("register_form"):
                if auth_mode == "supabase":
                    new_email = st.text_input("Email", key="reg_email")
                    new_pass = st.text_input("Password (min 6 characters)", type="password", key="reg_pass")
                    new_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2")
                    
                    if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                        if not all([new_email, new_pass, new_pass2]):
                            st.warning("Please fill all fields")
                        elif new_pass != new_pass2:
                            st.error("Passwords don't match")
                        elif len(new_pass) < 6:
                            st.error("Password must be at least 6 characters")
                        else:
                            success, msg, user_data = register_user_supabase(new_email, new_pass)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
                else:
                    new_user = st.text_input("Username", key="reg_user")
                    new_email = st.text_input("Email", key="reg_email_local")
                    new_pass = st.text_input("Password", type="password", key="reg_pass_local")
                    new_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2_local")
                    
                    if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                        if not all([new_user, new_email, new_pass, new_pass2]):
                            st.warning("Please fill all fields")
                        elif new_pass != new_pass2:
                            st.error("Passwords don't match")
                        else:
                            success, msg = register_user_local(new_user, new_email, new_pass)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
        
        with tab3:
            if auth_mode == "supabase":
                with st.form("reset_form"):
                    reset_email = st.text_input("Email", key="reset_email")
                    
                    if st.form_submit_button("Send Reset Link", use_container_width=True):
                        if reset_email:
                            success, msg = reset_password(reset_email)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
                        else:
                            st.warning("Please enter your email")
            else:
                st.info("Password reset is only available with Supabase authentication.")


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
    if MODEL_COMPARISON_PATH.exists():
        MODEL_COMPARISON_PATH.unlink()
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
        if MODEL_COMPARISON_PATH.exists():
            MODEL_COMPARISON_PATH.unlink()
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
                    if MODEL_COMPARISON_PATH.exists():
                        MODEL_COMPARISON_PATH.unlink()
                    st.session_state.clinical_chat_messages = None
                    st.rerun()
            with col2:
                if st.button("🔄 Retrain", use_container_width=True):
                    if METRICS_PATH.exists():
                        METRICS_PATH.unlink()
                    if MODEL_COMPARISON_PATH.exists():
                        MODEL_COMPARISON_PATH.unlink()
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
        if MODEL_COMPARISON_PATH.exists():
            MODEL_COMPARISON_PATH.unlink()
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
        
        with st.spinner("Running federated training with XGBoost, Random Forest & LightGBM (20 rounds × 3 clients)..."):
            progress.progress(10, "Training XGBoost...")
            progress.progress(40, "Training Random Forest...")
            progress.progress(70, "Training LightGBM...")
            metrics = run_federated_training(num_rounds=20, num_clients=3, local_epochs=10)
            progress.progress(100, "Complete!")
        
        st.success(f"✓ Training complete! All 3 models trained successfully.")
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

# Load model comparison for best model info
best_model_name = "XGBoost"
best_accuracy = latest_acc
if MODEL_COMPARISON_PATH.exists():
    with open(MODEL_COMPARISON_PATH) as f:
        _model_comp = json.load(f)
    best_key = _model_comp.get("best_model", "xgboost")
    best_model_name = _model_comp[best_key]["name"]
    best_accuracy = _model_comp[best_key]["final_accuracy"]

# KPI Cards
st.header("📈 Performance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    delta = (accuracies[-1] - accuracies[0]) * 100 if len(accuracies) > 1 else 0
    st.markdown(f"""
    <div class="metric-card green">
        <div class="metric-label">Best Accuracy</div>
        <div class="metric-value">{best_accuracy*100:.1f}%</div>
        <div class="metric-label">{best_model_name}</div>
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
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Models Trained</div>
        <div class="metric-value">3</div>
        <div class="metric-label">XGB / RF / LGB</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Model Comparison Section
# =============================================================================
st.header("🔬 Model Comparison")

if MODEL_COMPARISON_PATH.exists():
    with open(MODEL_COMPARISON_PATH) as f:
        model_comparison = json.load(f)
    
    # Model comparison cards
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    models_info = [
        ("xgboost", "XGBoost", "🚀", comp_col1),
        ("random_forest", "Random Forest", "🌲", comp_col2),
        ("lightgbm", "LightGBM", "⚡", comp_col3),
    ]
    
    best_model_key = model_comparison.get("best_model", "xgboost")
    
    for model_key, model_name, emoji, col in models_info:
        model_data = model_comparison.get(model_key, {})
        acc = model_data.get("final_accuracy", 0)
        loss = model_data.get("final_loss", 0)
        is_best = model_key == best_model_key
        
        with col:
            if is_best:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
                            border: 3px solid #FFD700; box-shadow: 0 4px 15px rgba(17,153,142,0.4);">
                    <div style="font-size: 2rem;">{emoji} 👑</div>
                    <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0;">{model_name}</div>
                    <div style="font-size: 1.8rem; font-weight: 700;">{acc*100:.2f}%</div>
                    <div style="font-size: 0.85rem; opacity: 0.9;">Loss: {loss:.4f}</div>
                    <div style="font-size: 0.75rem; margin-top: 0.5rem; font-weight: 600;">⭐ BEST MODEL</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
                    <div style="font-size: 2rem;">{emoji}</div>
                    <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0;">{model_name}</div>
                    <div style="font-size: 1.8rem; font-weight: 700;">{acc*100:.2f}%</div>
                    <div style="font-size: 0.85rem; opacity: 0.9;">Loss: {loss:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Summary table
    st.subheader("📊 Detailed Comparison")
    
    comparison_df = pd.DataFrame([
        {
            "Model": model_comparison[key]["name"],
            "Final Accuracy": f"{model_comparison[key]['final_accuracy']*100:.2f}%",
            "Final Loss": f"{model_comparison[key]['final_loss']:.4f}",
            "Status": "👑 Best" if key == best_model_key else ""
        }
        for key in ["xgboost", "random_forest", "lightgbm"]
    ])
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # All 6 Model Training Graphs (Accuracy & Loss for each model)
    # =========================================================================
    st.header("📊 Training History - All Models")
    
    # Define model configurations
    model_configs = [
        {
            "key": "xgboost",
            "name": "XGBoost",
            "emoji": "🚀",
            "acc_color": "#667eea",
            "loss_color": "#764ba2",
            "fill_acc": "rgba(102, 126, 234, 0.1)",
            "fill_loss": "rgba(118, 75, 162, 0.1)",
        },
        {
            "key": "random_forest",
            "name": "Random Forest",
            "emoji": "🌲",
            "acc_color": "#11998e",
            "loss_color": "#38ef7d",
            "fill_acc": "rgba(17, 153, 142, 0.1)",
            "fill_loss": "rgba(56, 239, 125, 0.1)",
        },
        {
            "key": "lightgbm",
            "name": "LightGBM",
            "emoji": "⚡",
            "acc_color": "#f2994a",
            "loss_color": "#f2c94c",
            "fill_acc": "rgba(242, 153, 74, 0.1)",
            "fill_loss": "rgba(242, 201, 76, 0.1)",
        },
    ]
    
    for config in model_configs:
        model_key = config["key"]
        model_data = model_comparison.get(model_key, {})
        round_metrics = model_data.get("round_metrics", [])
        
        if round_metrics:
            st.subheader(f"{config['emoji']} {config['name']} Training History")
            
            rounds = [m["round"] for m in round_metrics]
            accs = [m["accuracy"] * 100 for m in round_metrics]
            losses_data = [m.get("loss", 0) for m in round_metrics]
            
            chart_col1, chart_col2 = st.columns(2)
            
            # Accuracy Graph
            with chart_col1:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=rounds,
                    y=accs,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color=config["acc_color"], width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor=config["fill_acc"]
                ))
                fig_acc.update_layout(
                    title=f"{config['name']} - Accuracy per Round",
                    xaxis_title="Round",
                    yaxis_title="Accuracy (%)",
                    yaxis=dict(range=[0, 100]),
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            # Loss Graph
            with chart_col2:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=rounds,
                    y=losses_data,
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color=config["loss_color"], width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor=config["fill_loss"]
                ))
                fig_loss.update_layout(
                    title=f"{config['name']} - Loss per Round",
                    xaxis_title="Round",
                    yaxis_title="Loss",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig_loss, use_container_width=True)
    
    st.divider()
    
    # Combined comparison charts
    st.header("📈 Combined Model Comparison")
    
    combined_col1, combined_col2 = st.columns(2)
    
    with combined_col1:
        # All models accuracy comparison line chart
        fig_all_acc = go.Figure()
        
        for config in model_configs:
            model_key = config["key"]
            model_data = model_comparison.get(model_key, {})
            round_metrics = model_data.get("round_metrics", [])
            if round_metrics:
                rounds = [m["round"] for m in round_metrics]
                accs = [m["accuracy"] * 100 for m in round_metrics]
                fig_all_acc.add_trace(go.Scatter(
                    x=rounds,
                    y=accs,
                    mode='lines+markers',
                    name=config["name"],
                    line=dict(color=config["acc_color"], width=2),
                    marker=dict(size=6),
                ))
        
        fig_all_acc.update_layout(
            title="All Models - Accuracy Over Rounds",
            xaxis_title="Round",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_white",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_all_acc, use_container_width=True)
    
    with combined_col2:
        # All models loss comparison line chart
        fig_all_loss = go.Figure()
        
        for config in model_configs:
            model_key = config["key"]
            model_data = model_comparison.get(model_key, {})
            round_metrics = model_data.get("round_metrics", [])
            if round_metrics:
                rounds = [m["round"] for m in round_metrics]
                losses_data = [m.get("loss", 0) for m in round_metrics]
                fig_all_loss.add_trace(go.Scatter(
                    x=rounds,
                    y=losses_data,
                    mode='lines+markers',
                    name=config["name"],
                    line=dict(color=config["loss_color"], width=2),
                    marker=dict(size=6),
                ))
        
        fig_all_loss.update_layout(
            title="All Models - Loss Over Rounds",
            xaxis_title="Round",
            yaxis_title="Loss",
            template="plotly_white",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_all_loss, use_container_width=True)
    
    # Bar chart comparison
    bar_col1, bar_col2 = st.columns(2)
    
    with bar_col1:
        # Final accuracy bar chart
        model_names_list = [c["name"] for c in model_configs]
        final_accuracies = [
            model_comparison[c["key"]]["final_accuracy"] * 100
            for c in model_configs
        ]
        bar_colors = [c["acc_color"] for c in model_configs]
        
        fig_bar_acc = go.Figure(data=[
            go.Bar(
                x=model_names_list,
                y=final_accuracies,
                marker_color=bar_colors,
                text=[f"{a:.2f}%" for a in final_accuracies],
                textposition='auto',
            )
        ])
        fig_bar_acc.update_layout(
            title="Final Accuracy Comparison",
            xaxis_title="Model",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_bar_acc, use_container_width=True)
    
    with bar_col2:
        # Final loss bar chart
        final_losses = [
            model_comparison[c["key"]]["final_loss"]
            for c in model_configs
        ]
        loss_colors = [c["loss_color"] for c in model_configs]
        
        fig_bar_loss = go.Figure(data=[
            go.Bar(
                x=model_names_list,
                y=final_losses,
                marker_color=loss_colors,
                text=[f"{l:.4f}" for l in final_losses],
                textposition='auto',
            )
        ])
        fig_bar_loss.update_layout(
            title="Final Loss Comparison",
            xaxis_title="Model",
            yaxis_title="Loss",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_bar_loss, use_container_width=True)

else:
    st.info("Model comparison data will appear after training completes.")

st.divider()

# AI Insights - Multi-Model Summary
st.header("🤖 AI Clinical Insights")

if "clinical_chat_messages" not in st.session_state:
    st.session_state.clinical_chat_messages = None

_llm_data_context = get_training_data_context()
_dataset_info = get_dataset_info_for_display()

# Load model comparison for AI analysis
_ai_model_comparison = None
if MODEL_COMPARISON_PATH.exists():
    with open(MODEL_COMPARISON_PATH) as f:
        _ai_model_comparison = json.load(f)

if st.session_state.clinical_chat_messages is None:
    # Show prediction task context
    if _dataset_info:
        st.markdown("### 🎯 Prediction Task")
        
        task_col1, task_col2 = st.columns(2)
        with task_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem; border-radius: 10px; color: white;">
                <div style="font-size: 0.85rem; opacity: 0.9;">PREDICTING</div>
                <div style="font-size: 1.3rem; font-weight: 600;">{_dataset_info['target_column']}</div>
                <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                    {_dataset_info['n_classes']} possible outcomes
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with task_col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        padding: 1rem; border-radius: 10px; color: white;">
                <div style="font-size: 0.85rem; opacity: 0.9;">DATASET</div>
                <div style="font-size: 1.3rem; font-weight: 600;">{_dataset_info['dataset_name']}</div>
                <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                    {_dataset_info['n_samples']:,} samples, {_dataset_info['n_features']} features
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show outcome classes
        st.markdown("**Outcome Classes:**")
        class_cols = st.columns(len(_dataset_info['class_names']))
        for i, (class_name, col) in enumerate(zip(_dataset_info['class_names'], class_cols)):
            count = _dataset_info['class_counts'].get(class_name, 0)
            pct = (count / _dataset_info['n_samples']) * 100 if _dataset_info['n_samples'] > 0 else 0
            with col:
                st.metric(label=str(class_name), value=f"{count:,}", delta=f"{pct:.1f}%")
        
        st.markdown("---")
    
    # Show model summary before generating AI insight
    if _ai_model_comparison:
        st.markdown("### 📊 Model Performance Summary")
        summary_cols = st.columns(3)
        model_info = [
            ("🚀 XGBoost", "xgboost"),
            ("🌲 Random Forest", "random_forest"),
            ("⚡ LightGBM", "lightgbm"),
        ]
        for (label, key), col in zip(model_info, summary_cols):
            data = _ai_model_comparison.get(key, {})
            acc = data.get("final_accuracy", 0) * 100
            is_best = key == _ai_model_comparison.get("best_model")
            with col:
                if is_best:
                    st.success(f"{label}: **{acc:.2f}%** 👑")
                else:
                    st.info(f"{label}: **{acc:.2f}%**")
    
    st.markdown("")
    if st.button("✨ Generate AI Insight for All Models", type="primary"):
        with st.spinner("Analyzing all 3 models..."):
            try:
                if _ai_model_comparison:
                    msgs, err = generate_multimodel_insight_with_history(
                        model_comparison=_ai_model_comparison,
                        num_rounds=n_rounds,
                        data_context=_llm_data_context,
                        dataset_info=_dataset_info,
                    )
                else:
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
