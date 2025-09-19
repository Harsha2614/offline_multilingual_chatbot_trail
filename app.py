import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import whisper
import tempfile

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
    lower = t.lower()
    eng_markers = [" how ", " what ", " why ", " when ", " where ", " who ",
                   " do ", " does ", " is ", " are ", " can "]
    if any(m in (" " + lower + " ") for m in eng_markers):
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
    import numpy as np
    import soundfile as sf
    import librosa

    # Decode mic recording into numpy
    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    # Ensure mono and 16kHz for Whisper
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if samplerate != 16000:
        data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)

    # Run Whisper directly on numpy array
    whisper_model = load_whisper()
    result = whisper_model.transcribe(data, fp16=False, language="en")

    raw_text = result.get("text", "").strip()
    english_text = translate(raw_text, "en")   # force translation to English
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
    st.set_page_config(page_title="Offline Multilingual Chatbot", page_icon="🛟")
    st.title("🛟 Offline Multilingual Disaster Chatbot")
    st.caption("Speak or type in any language — input will be converted to English for processing.")

    if st.button("🔄 Rebuild Knowledge Base"):
        rebuild_kb()

    # 🎤 Mic recorder
    audio_bytes = st_audiorec()

    # If mic audio exists, transcribe and store into session_state
    if audio_bytes is not None:
        try:
            recognized = transcribe_audio_bytes(audio_bytes)
            if recognized:
                st.session_state["input_box"] = recognized
                st.success(f"🗣️ Recognized (English): {recognized}")
        except Exception as e:
            st.error(f"Audio transcription failed: {e}")

    # Text input (prefill with recognized speech if available)
    user_text = st.text_input(
        "Type or speak your question (English only after processing):",
        value=st.session_state.get("input_box", ""),
        key="input_box"
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    send_enabled = bool(user_text.strip())

    if st.button("Send", disabled=not send_enabled):
        lang = safe_detect(user_text)
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

        final_answer = answer_en  # ✅ keep answers in English only

        st.session_state.history.append(("You", user_text, "en"))
        st.session_state.history.append(("Bot", final_answer, "en"))

        st.session_state["input_box"] = ""
        st.experimental_rerun()

    # Chat history
    for speaker, text, lang in st.session_state.history[-20:]:
        st.markdown(f"**{speaker} ({lang})**: {text}")


if __name__ == "__main__":
    main()
