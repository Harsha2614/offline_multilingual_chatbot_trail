# Offline Multilingual Disaster Chatbot

A multilingual FAQ chatbot for disaster response. Works offline once models are downloaded.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate   # (Windows)
source .venv/bin/activate # (Linux/Mac)

pip install -r requirements.txt
python models_download.py
python build_index.py --kb knowledge_base.csv --out artifacts
streamlit run app.py
