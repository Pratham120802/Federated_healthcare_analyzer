import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import json
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from federated.data_loader import (
    dataset_exists,
    delete_custom_dataset,
    get_dataset_summary,
    save_custom_dataset,
    get_column_info,
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
    quick_chat,
    stream_initial_insight,
    build_initial_clinical_messages,
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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main > div { padding-top: 2rem; }

    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-card.green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 4px 20px rgba(17, 153, 142, 0.3);
    }
    .kpi-card.orange {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        box-shadow: 0 4px 20px rgba(242, 153, 74, 0.3);
    }
    .kpi-card.blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        box-shadow: 0 4px 20px rgba(33, 147, 176, 0.3);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.25rem 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
    }
    .kpi-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        color: #333;
    }

    .insight-box {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        line-height: 1.6;
        color: #333;
    }

    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-good { background: #d4edda; color: #155724; }
    .badge-warn { background: #fff3cd; color: #856404; }
    .badge-info { background: #d1ecf1; color: #0c5460; }

    .login-container {
        max-width: 400px;
        margin: 8rem auto;
        padding: 2.5rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.2);
    }
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .login-header p {
        color: #666;
        font-size: 0.95rem;
    }
    .login-icon {
        font-size: 4rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .google-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 0.95rem;
        color: #333;
        transition: background 0.2s, box-shadow 0.2s;
        width: 100%;
        text-decoration: none;
    }
    .google-btn:hover {
        background: #f8f9fa;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .divider {
        display: flex;
        align-items: center;
        margin: 1.5rem 0;
    }
    .divider::before, .divider::after {
        content: "";
        flex: 1;
        border-bottom: 1px solid #ddd;
    }
    .divider span {
        padding: 0 1rem;
        color: #888;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
from dashboard.auth import check_login, register_user, register_google_user, get_user_info

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")


def show_login_page():
    """Display the login/register form with Google OAuth."""
    
    # Check for Google OAuth callback
    query_params = st.query_params
    if "code" in query_params and "state" in query_params:
        handle_google_callback(query_params.get("code"), query_params.get("state"))
        return
    
    st.markdown("""
    <div class="login-container">
        <div class="login-header">
            <div class="login-icon">🏥</div>
            <h1>Federated Healthcare</h1>
            <p>Secure login to access the analyzer</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Google Sign In Button
        if GOOGLE_CLIENT_ID:
            google_auth_url = get_google_auth_url()
            st.markdown(
                f'<a href="{google_auth_url}" class="google-btn">'
                '<svg width="18" height="18" viewBox="0 0 18 18"><path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z"/><path fill="#34A853" d="M9.003 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332C2.438 15.983 5.482 18 9.003 18z"/><path fill="#FBBC05" d="M3.964 10.712c-.18-.54-.282-1.117-.282-1.71 0-.593.102-1.17.282-1.71V4.96H.957C.347 6.175 0 7.55 0 9.002c0 1.452.348 2.827.957 4.042l3.007-2.332z"/><path fill="#EA4335" d="M9.003 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.464.891 11.428 0 9.002 0 5.48 0 2.438 2.017.957 4.958L3.964 7.29c.708-2.127 2.692-3.71 5.036-3.71z"/></svg>'
                'Continue with Google</a>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="divider"><span>or</span></div>', unsafe_allow_html=True)
        
        # Tab selection for Login/Register
        if "auth_tab" not in st.session_state:
            st.session_state.auth_tab = "login"
        
        tab1, tab2 = st.tabs(["Sign In", "Register"])
        
        with tab1:
            show_login_form()
        
        with tab2:
            show_register_form()
        
        st.markdown("---")
        st.markdown(
            "<p style='text-align:center;color:#888;font-size:0.85rem'>"
            "Demo credentials:<br>"
            "<code>admin / admin123</code>"
            "</p>",
            unsafe_allow_html=True
        )


def show_login_form():
    """Display the login form."""
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Enter username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")
        
        submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)
        
        if submit:
            if username and password:
                if check_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.warning("Please enter both username and password")


def show_register_form():
    """Display the registration form."""
    with st.form("register_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Choose a username (min 3 chars)", key="reg_user")
        email = st.text_input("Email", placeholder="Enter your email", key="reg_email")
        password = st.text_input("Password", type="password", placeholder="Choose a password (min 6 chars)", key="reg_pass")
        password2 = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="reg_pass2")
        
        submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        
        if submit:
            if not all([username, email, password, password2]):
                st.warning("Please fill in all fields")
            elif password != password2:
                st.error("Passwords do not match")
            else:
                success, message = register_user(username, email, password)
                if success:
                    st.success(message)
                    st.info("You can now sign in with your new account!")
                else:
                    st.error(message)


def get_google_auth_url() -> str:
    """Generate Google OAuth authorization URL."""
    import urllib.parse
    import secrets
    
    state = secrets.token_urlsafe(16)
    st.session_state.oauth_state = state
    
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "select_account"
    }
    
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"


def handle_google_callback(code: str, state: str):
    """Handle Google OAuth callback."""
    import requests
    
    # Verify state
    if state != st.session_state.get("oauth_state"):
        st.error("Invalid OAuth state. Please try again.")
        st.query_params.clear()
        return
    
    try:
        # Exchange code for tokens
        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": GOOGLE_REDIRECT_URI
            },
            timeout=10
        )
        
        if token_response.status_code != 200:
            st.error("Failed to authenticate with Google. Please try again.")
            st.query_params.clear()
            return
        
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        
        # Get user info
        user_response = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10
        )
        
        if user_response.status_code != 200:
            st.error("Failed to get user info from Google.")
            st.query_params.clear()
            return
        
        user_info = user_response.json()
        email = user_info.get("email", "")
        name = user_info.get("name", "")
        google_id = user_info.get("id", "")
        
        # Register or login user
        success, username = register_google_user(email, name, google_id)
        
        if success:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.auth_type = "google"
            st.query_params.clear()
            st.rerun()
        else:
            st.error("Failed to create account. Please try again.")
            st.query_params.clear()
            
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        st.query_params.clear()


# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Show login page if not authenticated
if not st.session_state.logged_in:
    show_login_page()
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏥 FL Healthcare")
    
    # User info and logout
    st.write(f"Logged in as: **{st.session_state.username}**")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
    
    st.markdown("---")
    
    # Dataset upload
    st.markdown("### Upload Dataset")
    supported_extensions = list(SUPPORTED_FORMATS.keys())
    st.caption(f"Formats: {', '.join(supported_extensions).upper()}")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=supported_extensions,
        key="dataset_upload"
    )
    
    if uploaded_file is not None:
        try:
            df = read_uploaded_file(uploaded_file)
            original_filename = uploaded_file.name
            st.success(f"Loaded: {len(df)} rows, {len(df.columns)} cols")
            
            with st.expander("Preview data"):
                st.dataframe(df.head(3))
            
            # Target column selection
            columns = df.columns.tolist()
            
            target_col = st.selectbox(
                "Select target column",
                columns,
                index=len(columns) - 1,
            )
            
            # Show target info
            target_nulls = df[target_col].isna().sum()
            target_unique = df[target_col].nunique()
            st.caption(f"Classes: {target_unique} unique values")
            if target_nulls > 0:
                st.warning(f"{target_nulls} null rows excluded")
            
            if st.button("Train on this dataset", type="primary"):
                save_custom_dataset(df, original_filename=original_filename)
                set_data_config(target_column=target_col)
                if METRICS_PATH.exists():
                    METRICS_PATH.unlink()
                st.session_state.clinical_chat_messages = None
                st.session_state.training_done = False
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    # Show active dataset info
    if dataset_exists():
        try:
            ds_summary = get_dataset_summary()
            st.markdown("---")
            st.markdown("### Active Dataset")
            st.write(f"**{ds_summary['name']}**")
            st.caption(
                f"Target: {ds_summary['target_column']}\n\n"
                f"{ds_summary['n_samples']} samples, {ds_summary['n_features']} features, "
                f"{ds_summary['n_classes']} classes"
            )
            if st.button("Remove dataset"):
                delete_custom_dataset()
                if METRICS_PATH.exists():
                    METRICS_PATH.unlink()
                st.session_state.clinical_chat_messages = None
                st.rerun()
        except Exception:
            pass

    st.markdown("---")
    st.caption("Federated Healthcare Analyzer v2.0")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;background:linear-gradient(135deg,#667eea,#764ba2);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "font-size:2.5rem;margin-bottom:0.25rem'>"
    "Federated Healthcare Analyzer</h1>"
    "<p style='text-align:center;color:#888;margin-bottom:2rem'>"
    "Train and analyze healthcare classification models with federated learning</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Check dataset uploaded
# ---------------------------------------------------------------------------
if not dataset_exists():
    st.markdown(
        "<div style='text-align:center;padding:6rem 2rem'>"
        "<h2 style='color:#667eea'>Upload Your Dataset</h2>"
        "<p style='color:#666;max-width:500px;margin:auto'>"
        "Use the sidebar to upload your disease dataset (CSV, Excel, JSON, Parquet).<br><br>"
        "The system will automatically run 5 rounds of federated training "
        "across 3 simulated hospital clients and show you the results."
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Run training if needed
# ---------------------------------------------------------------------------
if not METRICS_PATH.exists():
    st.markdown("<div class='section-header'>Training in Progress...</div>", unsafe_allow_html=True)
    
    progress_bar = st.progress(0, text="Initializing federated training...")
    status_text = st.empty()
    
    try:
        ds_summary = get_dataset_summary()
        status_text.markdown(
            f"**Dataset:** {ds_summary['n_samples']} samples, "
            f"{ds_summary['n_features']} features, {ds_summary['n_classes']} classes"
        )
        
        progress_bar.progress(10, text="Preparing client data partitions...")
        
        with st.spinner("Running 5 rounds of federated training across 3 clients..."):
            metrics = run_federated_training(num_rounds=5, num_clients=3, local_epochs=5)
        
        progress_bar.progress(100, text="Training complete!")
        st.success(f"Federated training complete! {len(metrics)} rounds finished.")
        st.rerun()
        
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------
if not METRICS_PATH.exists():
    st.warning("No training metrics found. Please upload a dataset and train.")
    st.stop()

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

if not metrics:
    st.warning("Metrics file is empty. Please re-train.")
    st.stop()

total_rounds = len(metrics)
accuracies = [m["accuracy"] for m in metrics]
losses = [m.get("loss", 1 - m["accuracy"]) for m in metrics]
latest_acc = accuracies[-1]
latest_loss = losses[-1]

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>Performance Overview</div>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    acc_delta = accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
    st.markdown(
        f"<div class='kpi-card green'><div class='kpi-label'>Accuracy</div>"
        f"<div class='kpi-value'>{latest_acc*100:.1f}%</div>"
        f"<div class='kpi-delta'>{'↑' if acc_delta >= 0 else '↓'} {abs(acc_delta)*100:.1f}pp</div></div>",
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"<div class='kpi-card orange'><div class='kpi-label'>Loss</div>"
        f"<div class='kpi-value'>{latest_loss:.4f}</div>"
        f"<div class='kpi-delta'>Lower is better</div></div>",
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"<div class='kpi-card blue'><div class='kpi-label'>Rounds</div>"
        f"<div class='kpi-value'>{total_rounds}</div>"
        f"<div class='kpi-delta'>Completed</div></div>",
        unsafe_allow_html=True,
    )

with kpi4:
    num_clients = metrics[-1].get("num_clients", 3) if metrics else 3
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Clients</div>"
        f"<div class='kpi-value'>{num_clients}</div>"
        f"<div class='kpi-delta'>Hospitals</div></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>Training Progress</div>", unsafe_allow_html=True)

chart1, chart2 = st.columns(2)

with chart1:
    fig_acc = go.Figure()
    fig_acc.add_trace(
        go.Scatter(
            x=list(range(1, total_rounds + 1)),
            y=[a * 100 for a in accuracies],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#11998e", width=3),
            marker=dict(size=10),
            fill="tozeroy",
            fillcolor="rgba(17, 153, 142, 0.1)",
        )
    )
    fig_acc.update_layout(
        title="Accuracy per Round",
        xaxis_title="Round",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_acc, use_container_width=True)

with chart2:
    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(
            x=list(range(1, total_rounds + 1)),
            y=losses,
            mode="lines+markers",
            name="Loss",
            line=dict(color="#f2994a", width=3),
            marker=dict(size=10),
            fill="tozeroy",
            fillcolor="rgba(242, 153, 74, 0.1)",
        )
    )
    fig_loss.update_layout(
        title="Loss per Round",
        xaxis_title="Round",
        yaxis_title="Loss",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_loss, use_container_width=True)

# ---------------------------------------------------------------------------
# Round details
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>Round Details</div>", unsafe_allow_html=True)

round_data = []
for i, m in enumerate(metrics, 1):
    round_data.append({
        "Round": i,
        "Accuracy": f"{m['accuracy']*100:.2f}%",
        "Loss": f"{m.get('loss', 1-m['accuracy']):.4f}",
        "Clients": m.get("num_clients", 3),
        "Samples": m.get("total_examples", "N/A"),
    })

st.dataframe(
    round_data,
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# LLM Insights
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>AI Clinical Insight</div>", unsafe_allow_html=True)

_llm_data_context = get_training_data_context()

if "clinical_chat_messages" not in st.session_state:
    st.session_state.clinical_chat_messages = None
if "clinical_chat_fp" not in st.session_state:
    st.session_state.clinical_chat_fp = None

metrics_fp = (round(latest_acc, 6), total_rounds)

btn_row1, btn_row2 = st.columns([2, 1])
with btn_row1:
    if st.session_state.clinical_chat_messages is None:
        if st.button("Generate Clinical Insight", type="primary", use_container_width=True):
            insight_container = st.empty()
            with insight_container.container():
                st.markdown("**AI Insight:**")
                response_area = st.empty()
                full_response = ""
                final_msgs = None
                
                for chunk, msgs in stream_initial_insight(
                    accuracy=latest_acc,
                    loss=latest_loss,
                    num_rounds=total_rounds,
                    accuracy_history=accuracies,
                    data_context=_llm_data_context,
                ):
                    full_response += chunk
                    response_area.markdown(full_response + "▌")
                    if msgs:
                        final_msgs = msgs
                
                response_area.markdown(full_response)
            
            if final_msgs:
                st.session_state.clinical_chat_messages = final_msgs
                st.session_state.clinical_chat_fp = metrics_fp
                st.rerun()
            elif full_response:
                st.error("Failed to generate insight. Check Ollama is running.")
    else:
        regen, clear = st.columns(2)
        with regen:
            if st.button("Regenerate insight", use_container_width=True):
                insight_container = st.empty()
                with insight_container.container():
                    response_area = st.empty()
                    full_response = ""
                    final_msgs = None
                    
                    for chunk, msgs in stream_initial_insight(
                        accuracy=latest_acc,
                        loss=latest_loss,
                        num_rounds=total_rounds,
                        accuracy_history=accuracies,
                        data_context=_llm_data_context,
                    ):
                        full_response += chunk
                        response_area.markdown(full_response + "▌")
                        if msgs:
                            final_msgs = msgs
                    
                    response_area.markdown(full_response)
                
                if final_msgs:
                    st.session_state.clinical_chat_messages = final_msgs
                    st.rerun()
        with clear:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.clinical_chat_messages = None
                st.rerun()

# Display chat
if st.session_state.clinical_chat_messages:
    with st.container():
        chat_box = st.container()
    with chat_box:
        st.caption("Conversation — ask follow-up questions about your results")
        for msg in st.session_state.clinical_chat_messages:
            role = msg.get("role", "")
            if role == "system":
                continue
            with st.chat_message(role):
                st.markdown(msg.get("content", ""))

    if follow_up := st.chat_input(
        "Ask a follow-up question...",
        key="clinical_follow_up",
    ):
        st.session_state.clinical_chat_messages.append({"role": "user", "content": follow_up})
        
        with st.chat_message("user"):
            st.markdown(follow_up)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in ollama_chat_stream(st.session_state.clinical_chat_messages, max_tokens=150):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        if full_response and "[Error:" not in full_response:
            st.session_state.clinical_chat_messages.append(
                {"role": "assistant", "content": full_response}
            )
        else:
            st.session_state.clinical_chat_messages.pop()
            if "[Error:" in full_response:
                st.error(full_response)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.8rem;padding:1rem 0'>"
    "Federated Healthcare Analyzer — Powered by Flower, PyTorch & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
