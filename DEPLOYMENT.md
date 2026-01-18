# Deployment Guide: SentinelMD on Streamlit Cloud

This guide explains how to deploy the **SentinelMD** frontend to Streamlit Cloud while keeping your private **MedGemma** model running on your local machine (using a secure tunnel).

## Prerequisites
1.  **Streamlit Cloud Account**
2.  **Ngrok Account**
3.  **Local Machine** with Ollama running

---

## Step 1: Start Local Services

1.  **Start Ollama**:
    ```bash
    ollama serve
    ```

2.  **Start Ngrok Tunnel**:
    Open a *new* terminal window and run:
    ```bash
    ngrok http 11434 --host-header="localhost:11434"
    ```
    *Note: The `--host-header` flag is crucial for Ollama security checks.*

3.  **Copy the URL**:
    Ngrok will display a Forwarding URL, e.g., `https://a1b2-c3d4.ngrok-free.app`. **Copy this URL.**

---

## Step 2: Push Code to GitHub

Ensure your latest changes (including `requirements.txt`) are pushed:
```bash
git add .
git commit -m "Prepare for deployment"
git push
```

---

## Step 3: Deploy on Streamlit Cloud

1.  Go to **[share.streamlit.io](https://share.streamlit.io)**.
2.  Click **New App**.
3.  Select your repository (`Magma4/sentinel`), branch (`main`), and file (`src/app/ui_streamlit.py`).
4.  Click **Deploy**.

---

## Step 4: Configure Secrets (The Magic Step 🪄)

Once the app is deploying (or if it fails initially), you need to tell it where to find your model:

1.  In your Streamlit App, click the **Settings (three dots)** in the top right -> **Settings**.
2.  Go to **Secrets** on the left.
3.  Paste the following (replace with your actuall ngrok URL):

    ```toml
    OLLAMA_HOST = "https://a1b2-c3d4.ngrok-free.app"
    ```

4.  Click **Save**. The app should auto-reload.

---
