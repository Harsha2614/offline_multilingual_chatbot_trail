from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from utils import TRANSLATION_MODEL_ID

def main():
    # Embedding model
    SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Download + cache translation model once
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_ID)

    print("✅ NLLB-200 model downloaded and ready.")

if __name__ == "__main__":
    main()
