import os
import io
import re
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import whisper
import warnings
import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from st_audiorec import st_audiorec
from utils import detect_lang, LANG_CODE_MAP, TRANSLATION_MODEL_ID

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch").setLevel(logging.ERROR)

# ---------------- Paths ----------------
ART_DIR = "artifacts"
KB_SOURCE = "knowledge_base.csv"
KB_CSV = os.path.join(ART_DIR, "kb_rows.csv")
KB_EMB = os.path.join(ART_DIR, "kb_embeddings.npy")
META_FILE = os.path.join(ART_DIR, "kb_meta.txt")

# ---------------- Cached Loaders ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_kb():
    if os.path.exists(KB_CSV) and os.path.exists(KB_EMB):
        df = pd.read_csv(KB_CSV)
        embeddings = np.load(KB_EMB)
        return df, embeddings
    return None, None

@st.cache_resource
def load_translator():
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_ID)
    return tokenizer, model

@st.cache_resource
def load_general_qa():
    return pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# ---------------- Helper Functions ----------------
def safe_detect(text: str) -> str:
    if not text or len(text.strip()) < 2:
        return "en"
    lang = detect_lang(text)
    return lang if lang in LANG_CODE_MAP else "en"

def translate(text: str, target_lang: str = "en") -> str:
    tokenizer, model = load_translator()
    src = safe_detect(text)
    if src == target_lang:
        return text
    src_lang = LANG_CODE_MAP.get(src, "eng_Latn")
    tgt_lang = LANG_CODE_MAP.get(target_lang, "eng_Latn")
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    generated = model.generate(**encoded, forced_bos_token_id=forced_bos_token_id, max_length=256)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    import soundfile as sf
    import librosa
    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if samplerate != 16000:
        data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
    result = load_whisper().transcribe(data, fp16=False)
    return result.get("text", "").strip()

# ---------------- KB Management ----------------
def preprocess_query(text: str) -> str:
    """Normalize user query before embedding."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

def cosine_search(embedder, kb_embeddings, df, query: str, k: int = 3):
    query = preprocess_query(query)
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = np.dot(kb_embeddings, q_emb.T).flatten()
    topk_idx = sims.argsort()[::-1][:k]
    if len(topk_idx) == 0:
        return None
    best_idx = topk_idx[0]
    score = sims[best_idx]
    if score < 0.45:  # soft threshold
        return None
    row = df.iloc[best_idx]
    return {"id": row["id"], "score": float(score), "answer_en": row["answer_en"]}

def rebuild_kb():
    """Rebuilds bilingual knowledge base embeddings."""
    st.info("🔄 Rebuilding Knowledge Base... please wait.")
    try:
        os.makedirs(ART_DIR, exist_ok=True)
        df = pd.read_csv(KB_SOURCE)

        required_cols = {"id", "question_en", "answer_en", "question_te", "answer_te"}
        if not required_cols.issubset(df.columns):
            st.error("❌ 'knowledge_base.csv' must include: id, question_en, answer_en, question_te, answer_te")
            return

        embedder = load_embedder()
        bilingual_texts = (
            df["question_en"].fillna("") + " " + df["answer_en"].fillna("") + " " +
            df["question_te"].fillna("") + " " + df["answer_te"].fillna("")
        ).tolist()

        progress_bar = st.progress(0)
        embeddings = []
        for i, t in enumerate(bilingual_texts):
            emb = embedder.encode(t, convert_to_numpy=True, normalize_embeddings=True)
            embeddings.append(emb)
            progress_bar.progress((i + 1) / len(bilingual_texts))

        embeddings = np.array(embeddings)
        np.save(KB_EMB, embeddings)
        df.to_csv(KB_CSV, index=False)

        with open(META_FILE, "w") as f:
            f.write(f"Last rebuilt: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total entries: {len(df)}")

        st.success(f"✅ Bilingual knowledge base rebuilt successfully with {len(df)} entries!")
        st.session_state["kb_last_rebuilt"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["kb_entries"] = len(df)
        st.balloons()

    except Exception as e:
        if "RerunData" not in str(e):
            st.error(f"⚠️ Error rebuilding KB: {e}")

# ---------------- Chatbot UI ----------------
def main():
    st.set_page_config(page_title="Offline Multilingual Chatbot", page_icon="💬", layout="wide")

    # --- Custom CSS (UI styling) ---
    st.markdown("""
        <style>
        .chat-btn {
            position: fixed;
            bottom: 25px;
            right: 25px;
            background-color: #0084FF;
            color: white;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 28px;
            cursor: pointer;
            z-index: 9999;
            border: none;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            transition: transform 0.2s ease-in-out;
        }
        .chat-btn:hover { transform: scale(1.1); }

        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 1rem;
            border-radius: 10px;
            background-color: #f9f9f9;
            margin-bottom: 1rem;
            animation: slideUp 0.5s ease-out;
        }
        .user-msg {
            background-color: #DCF8C6;
            padding: 10px 14px;
            border-radius: 18px 18px 0 18px;
            margin: 5px;
            display: inline-block;
            max-width: 70%;
            word-wrap: break-word;
            text-align: left;
            align-self: flex-end;
            animation: fadeIn 0.3s ease-in-out;
        }
        .bot-msg {
            background-color: #E5E5EA;
            padding: 10px 14px;
            border-radius: 18px 18px 18px 0;
            margin: 5px;
            display: inline-block;
            max-width: 70%;
            word-wrap: break-word;
            text-align: left;
            align-self: flex-start;
            animation: fadeIn 0.3s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("💬 Offline Multilingual Chatbot")

    # --- KB Management Section ---
    st.subheader("🗂 Knowledge Base Management")
    if st.button("🔄 Rebuild Knowledge Base"):
        rebuild_kb()

    if os.path.exists(META_FILE):
        with open(META_FILE) as f:
            st.caption(f.read())
    if "kb_last_rebuilt" in st.session_state:
        st.caption(f"🕒 Last rebuilt (session): {st.session_state['kb_last_rebuilt']} | Entries: {st.session_state.get('kb_entries', '?')}")

    st.divider()

    # --- Voice input ---
    audio_bytes = st_audiorec()
    if audio_bytes:
        try:
            recognized = transcribe_audio_bytes(audio_bytes)
            if recognized:
                st.session_state["recognized_text"] = recognized
                st.success(f"🎙 Recognized: {recognized}")
        except Exception as e:
            st.error(f"Audio error: {e}")

    # --- Text input ---
    user_text = st.text_input("Type or speak your question:", value=st.session_state.get("recognized_text", ""), key="input_text")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑 Clear Chat"):
            st.session_state.history = []
            st.session_state["recognized_text"] = ""
            st.rerun()
    with col2:
        send_pressed = st.button("📤 Send")

    if send_pressed and user_text.strip():
        lang = safe_detect(user_text)
        query_en = translate(user_text, "en")
        df, kb_embeddings = load_kb()
        answer_en = None

        if df is not None and kb_embeddings is not None:
            result = cosine_search(load_embedder(), kb_embeddings, df, query_en)
            if result:
                answer_en = result["answer_en"]

        if not answer_en:
            qa = load_general_qa()
            out = qa(f"Answer briefly:\n{query_en}", max_new_tokens=64, do_sample=False)
            answer_en = out[0]["generated_text"]

        final_answer = translate(answer_en, lang)
        st.session_state.history.append(("You", user_text, "user"))
        st.session_state.history.append(("Bot", final_answer, "bot"))
        st.session_state["recognized_text"] = ""
        st.rerun()

    # --- Chat Display ---
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for sender, msg, role in st.session_state.history[-20:]:
        style = "user-msg" if role == "user" else "bot-msg"
        st.markdown(f"<div class='{style}'>{msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
