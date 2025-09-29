import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import whisper

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from st_audiorec import st_audiorec  # 🎤 mic recorder

from utils import detect_lang, LANG_CODE_MAP, TRANSLATION_MODEL_ID

# ---------------- Paths ----------------
ART_DIR = "artifacts"
KB_CSV = os.path.join(ART_DIR, "kb_rows.csv")
KB_EMB = os.path.join(ART_DIR, "kb_embeddings.npy")
KB_SOURCE = "knowledge_base.csv"

# ---------------- Instant replies ----------------
INSTANT_REPLIES = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hello! I’m your offline assistant. Ask me anything.",
    "thanks": "You're welcome!",
    "thank you": "Glad to help!",
    "who are you": "I’m your offline multilingual disaster chatbot, here to help.",
    "how are you": "I’m running fine and ready to assist you.",
}

# ---------------- Cached loaders ----------------
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

# ---------------- Helpers ----------------
def safe_detect(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "en"
    if len(t) < 5:
        return "en"
    detected = detect_lang(text)
    if detected not in LANG_CODE_MAP:
        return "en"
    return detected

def get_instant_reply(text: str):
    text_lower = (text or "").lower().strip()
    for key, val in INSTANT_REPLIES.items():
        if key in text_lower:
            return val
    return None

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
    generated = model.generate(
        **encoded,
        forced_bos_token_id=forced_bos_token_id,
        max_length=256,
        do_sample=False,
        num_beams=2,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def cosine_search(embedder, kb_embeddings, df, query: str, k: int = 3):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = np.dot(kb_embeddings, q_emb.T).flatten()
    topk_idx = sims.argsort()[::-1][:k]
    results = []
    for i in topk_idx:
        row = df.iloc[i]
        results.append({"id": row["id"], "score": float(sims[i]), "answer_en": row["answer_en"]})
    return results

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Convert mic audio to English text using Whisper (without ffmpeg)."""
    import soundfile as sf
    import librosa

    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if samplerate != 16000:
        data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)

    whisper_model = load_whisper()
    result = whisper_model.transcribe(data, fp16=False, language="en")

    raw_text = result.get("text", "").strip()
    english_text = translate(raw_text, "en")  # force English
    return english_text

def rebuild_kb():
    os.makedirs(ART_DIR, exist_ok=True)
    df = pd.read_csv(KB_SOURCE)
    texts = (df["question_en"].fillna("") + " " + df["answer_en"].fillna("")).tolist()
    embedder = load_embedder()
    embeddings = embedder.encode(
        texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
    )
    np.save(KB_EMB, embeddings)
    df.to_csv(KB_CSV, index=False)
    st.success("✅ Knowledge Base rebuilt successfully!")

# ---------------- Main app ----------------
def main():
    st.set_page_config(page_title="Offline Multilingual Chatbot", page_icon="🛟", layout="wide")

    # --- Custom CSS for chatbot look ---
    st.markdown("""
        <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 1rem;
            border-radius: 10px;
            background-color: #f9f9f9;
            margin-bottom: 1rem;
        }
        .user-msg {
            background-color: #DCF8C6;
            color: black;
            padding: 10px 15px;
            border-radius: 20px;
            margin: 5px;
            text-align: right;
            display: inline-block;
            max-width: 80%;
        }
        .bot-msg {
            background-color: #E5E5EA;
            color: black;
            padding: 10px 15px;
            border-radius: 20px;
            margin: 5px;
            text-align: left;
            display: inline-block;
            max-width: 80%;
        }
        .chat-row {
            display: flex;
            margin-bottom: 10px;
        }
        .chat-row.user { justify-content: flex-end; }
        .chat-row.bot { justify-content: flex-start; }
        .mic-button {
            background: #007BFF;
            color: white;
            border-radius: 50%;
            padding: 12px;
            border: none;
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛟 Offline Multilingual Disaster Chatbot")

    if st.button("🔄 Rebuild Knowledge Base"):
        rebuild_kb()

    # 🎤 Mic recorder
    audio_bytes = st_audiorec()

    if audio_bytes is not None:
        try:
            recognized = transcribe_audio_bytes(audio_bytes)
            if recognized:
                st.session_state["recognized_text"] = recognized
                st.success(f"🗣️ Recognized (English): {recognized}")
        except Exception as e:
            st.error(f"Audio transcription failed: {e}")

    # Text input
    user_text = st.text_input(
        "Type or speak your question:",
        value=st.session_state.get("recognized_text", ""),
        key="input_text"
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Send") and user_text.strip():
        query_en = translate(user_text, "en")

        answer_en = get_instant_reply(query_en)
        if not answer_en:
            embedder = load_embedder()
            df, kb_embeddings = load_kb()
            if df is not None and kb_embeddings is not None:
                results = cosine_search(embedder, kb_embeddings, df, query_en, k=3)
                if results and results[0]["score"] > 0.55:
                    answer_en = results[0]["answer_en"]

        if not answer_en:
            qa = load_general_qa()
            prompt = f"Answer concisely and helpfully:\n\n{query_en}"
            out = qa(prompt, max_new_tokens=64, do_sample=False)
            answer_en = out[0].get("generated_text", "").strip()

        st.session_state.history.append(("You", user_text, "user"))
        st.session_state.history.append(("Bot", answer_en, "bot"))

        st.session_state["recognized_text"] = ""
        st.rerun()


    # --- Chat history container ---
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for speaker, text, role in st.session_state.history[-20:]:
        if role == "user":
            st.markdown(f"<div class='chat-row user'><div class='user-msg'>{text}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-row bot'><div class='bot-msg'>{text}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


