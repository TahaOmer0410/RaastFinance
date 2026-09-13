import faiss
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def chunk_text(text, chunk_size=120):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        words = p.split()
        if len(words) <= chunk_size:
            chunks.append(p)
        else:
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def build_index(kb_path):
    with open(kb_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    model = _get_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, chunks


def retrieve(index, chunks, query, top_k=3):
    if index is None or not chunks:
        raise RuntimeError("Index not built. Call build_index first.")
    model = _get_model()
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
