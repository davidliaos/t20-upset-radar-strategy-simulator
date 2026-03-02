# Deploy to Streamlit Community Cloud

## Prerequisites

1. **GitHub**: Push your code to a GitHub repository.
2. **Streamlit account**: Sign up at [share.streamlit.io](https://share.streamlit.io).

## Deployment Steps

1. **Commit and push** your code to GitHub (including `data/raw/world_cup_last_30_years.csv` and model artifacts in `models/`).

2. **Connect GitHub** to Streamlit Community Cloud:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in and connect your GitHub account

3. **Create app**:
   - Click **"Create app"** (upper right)
   - Choose **"Yup, I have an app"**
   - **Repository**: `your-username/T20-World-Cup-2026` (or your repo path)
   - **Branch**: `master` (or `main`)
   - **Main file path**: `app.py`
   - Optional: set a custom subdomain (e.g. `t20-upset-radar`)

4. **Advanced settings** (optional):
   - **Python version**: 3.11 or 3.12 (default)
   - No secrets required for this app

5. **Deploy**: Click **Deploy**. The app will build and launch in a few minutes.

## What Gets Deployed

- **Entrypoint**: `app.py` (Streamlit runs from repo root)
- **Dependencies**: `requirements.txt`
- **Data**: `data/raw/world_cup_last_30_years.csv` (must be in repo)
- **Model**: `models/baseline_logistic_calibrated.joblib` (for fast startup; if missing, app trains on first load)

## Troubleshooting

- **Slow first load**: If the model is not in the repo, the app trains on first run (~1–2 min). Include the model artifacts for faster startup.
- **Memory**: The app uses ~500MB–1GB. If you hit limits, reduce `@st.cache_data` usage or simplify the Insights tab.
- **Logs**: View deployment logs in the Streamlit Cloud dashboard (visible to repo collaborators).
