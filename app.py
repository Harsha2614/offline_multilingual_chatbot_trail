import os
import io
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

# ---------------- Helpers ----------------
def safe_detect(text: str) -> str:
    if not text or len(text.strip()) < 2:
        return "en"
    lang = detect_lang(text)
    return lang if lang in LANG_CODE_MAP else "en"

def get_instant_reply(text: str):
    text_lower = text.lower().strip()
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
    import soundfile as sf
    import librosa
    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if samplerate != 16000:
        data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
    result = load_whisper().transcribe(data, fp16=False)
    return result.get("text", "").strip()

# ---------------- Main ----------------
def main():
    st.set_page_config(page_title="Offline Multilingual Chatbot", page_icon="💬", layout="wide")

    # Session states
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "history" not in st.session_state:
        st.session_state.history = []

    # --- Floating Button ---
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
        }
        </style>
    """, unsafe_allow_html=True)

    # Show floating button always
    if not st.session_state.chat_open:
        if st.button("💬", key="open_chat", help="Open chatbot"):
            st.session_state.chat_open = True
            st.rerun()
        return

    # --- Chat UI ---
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
            padding: 8px 12px;
            border-radius: 18px;
            margin: 5px;
            text-align: right;
        }
        .bot-msg {
            background-color: #E5E5EA;
            padding: 8px 12px;
            border-radius: 18px;
            margin: 5px;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("💬 Offline Multilingual Chatbot")

    if st.button("❌ Close Chat"):
        st.session_state.chat_open = False
        st.rerun()

    audio_bytes = st_audiorec()
    if audio_bytes:
        try:
            recognized = transcribe_audio_bytes(audio_bytes)
            if recognized:
                st.session_state["recognized_text"] = recognized
                st.success(f"🎙 Recognized: {recognized}")
        except Exception as e:
            st.error(f"Audio error: {e}")

    user_text = st.text_input(
        "Type or speak your question:",
        value=st.session_state.get("recognized_text", ""),
        key="input_text"
    )

    if st.button("🗑 Clear Chat"):
        st.session_state.history = []
        st.session_state["recognized_text"] = ""
        st.rerun()

    if st.button("Send") and user_text.strip():
        lang = safe_detect(user_text)
        query_en = translate(user_text, "en")

        answer_en = get_instant_reply(query_en)
        if not answer_en:
            df, kb_embeddings = load_kb()
            if df is not None and kb_embeddings is not None:
                results = cosine_search(load_embedder(), kb_embeddings, df, query_en)
                if results and results[0]["score"] > 0.55:
                    answer_en = results[0]["answer_en"]

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
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for sender, msg, role in st.session_state.history[-20:]:
        style = "user-msg" if role == "user" else "bot-msg"
        st.markdown(f"<div class='{style}'>{msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
