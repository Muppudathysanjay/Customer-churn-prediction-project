# 🚀 Deployment Guide

This guide covers two steps: (1) publishing the project to GitHub, and (2) deploying the
live dashboard on Streamlit Community Cloud.

---

## 1. Deploying to GitHub

### Step 1 — Initialize the repository (if not already a git repo)
```bash
cd Customer-Churn-Prediction
git init
git add .
git commit -m "Initial commit: Customer Churn Prediction end-to-end project"
```

### Step 2 — Create a `.gitignore` (recommended)
```
__pycache__/
*.pyc
venv/
.ipynb_checkpoints/
.DS_Store
```

### Step 3 — Create a new GitHub repository
1. Go to [github.com/new](https://github.com/new)
2. Name it `Customer-Churn-Prediction`
3. Leave it empty (no README/license — you already have them locally)
4. Click **Create repository**

### Step 4 — Push your local project
```bash
git remote add origin https://github.com/<your-username>/Customer-Churn-Prediction.git
git branch -M main
git push -u origin main
```

Your project — code, notebook, dashboard, models, images, and reports — is now live on
GitHub and ready to be referenced in your resume or portfolio.

---

## 2. Deploying the Dashboard to Streamlit Community Cloud

Streamlit Community Cloud lets you deploy directly from a GitHub repository, for free.

### Step 1 — Confirm repo requirements
Streamlit Community Cloud needs:
- A `requirements.txt` at the repo root ✅ (already included)
- An entry-point Python file — this project uses `app.py` at the repo root ✅

### Step 2 — Sign in to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Authorize Streamlit to access your repositories

### Step 3 — Create a new app
1. Click **"New app"**
2. Select your `Customer-Churn-Prediction` repository
3. Set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy"**

### Step 4 — Wait for the build
Streamlit Cloud will install everything from `requirements.txt` and launch your app.
First-time builds typically take 2–5 minutes. You can watch the build logs live in the
deployment console.

### Step 5 — Verify the live app
Once deployed, you'll get a public URL like:
```
https://<your-username>-customer-churn-prediction.streamlit.app
```
Open it and confirm:
- KPI cards load correctly
- All four tabs (Overview, Predict, Performance, Insights) render without errors
- The prediction form returns a churn probability and risk gauge
- CSV downloads work

### Step 6 — Keep it updated
Any time you `git push` new changes to the `main` branch, Streamlit Community Cloud
automatically redeploys the app with the latest code.

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on deploy | Missing package in `requirements.txt` | Add the package and re-push |
| App can't find model files | Relative paths broken | Confirm `models/` and `data/` folders were committed (not gitignored) |
| Slow first load | Cold start + model loading | This is expected on free tier; subsequent loads are cached via `@st.cache_resource` |
| Plotly charts blank | Browser ad-blocker or CDN issue | Try a different browser or disable ad-blockers |

---

## Alternative Deployment Options

- **Hugging Face Spaces** — supports Streamlit apps directly via a `Dockerfile` or native Streamlit SDK
- **Render / Railway** — deploy as a standard web service using `streamlit run app.py --server.port $PORT`
- **Docker** — containerize with a simple Dockerfile:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY . .
  RUN pip install -r requirements.txt
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```
