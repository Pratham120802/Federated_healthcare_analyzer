# Federated Healthcare Analyzer

Upload your disease dataset and get complete federated learning analysis with AI-powered insights.

## How to Use

### 1. Start the dashboard
```bash
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

### 2. Login or Register
- **Login** with existing credentials (demo: `admin` / `admin123`)
- **Register** a new account
- **Google Sign In** (requires setup below)

### 3. Upload your dataset
- Upload a file in the sidebar (CSV, Excel, JSON, Parquet, TSV)
- Select the **target column** (what to predict)
- Click **"Train on this dataset"**

### 3. Automatic training
The system will automatically:
- Run **5 rounds** of federated learning
- Simulate **3 hospital clients** 
- Show accuracy and loss graphs
- Display round-by-round details

### 4. Get AI insights
- Click **"Generate Clinical Insight"**
- Ask follow-up questions about your results

## Supported File Formats

- **CSV** — Comma separated values
- **XLSX/XLS** — Excel files
- **JSON** — JSON records or lines
- **Parquet** — Apache Parquet
- **TSV/TXT** — Tab separated values

The system automatically handles:
- Categorical features (auto-encoded)
- Missing values (auto-filled)
- Target column selection

## Google OAuth Setup (Optional)

To enable Google Sign In:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Go to **APIs & Services > Credentials**
4. Click **Create Credentials > OAuth 2.0 Client ID**
5. Select **Web application**
6. Add authorized redirect URI: `http://localhost:8501`
7. Copy Client ID and Client Secret

Set environment variables:
```bash
set GOOGLE_CLIENT_ID=your_client_id_here
set GOOGLE_CLIENT_SECRET=your_client_secret_here
set GOOGLE_REDIRECT_URI=http://localhost:8501
```

Or create a `.env` file in the project root.

## AI Chat (Ollama)

For AI insights, run Ollama:
```bash
ollama serve
ollama pull qwen3:4b
```

Optional env vars:
- `OLLAMA_HOST` — default `http://localhost:11434`
- `OLLAMA_MODEL` — default `qwen3:4b`
