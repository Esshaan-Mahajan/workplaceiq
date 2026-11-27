import hashlib
import re
from dataclasses import dataclass

import numpy as np
import streamlit as st
from docx import Document
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="WorkplaceIQ", page_icon="🔎", layout="wide")

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"
RRF_K = 60


@dataclass
class Chunk:
    text: str
    filename: str
    number: int
    page: int | None = None

    @property
    def source(self):
        location = f", page {self.page}" if self.page else ""
        return f"{self.filename}{location}, chunk {self.number}"


def extract(file):
    ext = file.name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        reader = PdfReader(file)
        return [(p.extract_text() or "", i + 1) for i, p in enumerate(reader.pages)]
    if ext == "docx":
        doc = Document(file)
        return [("\n".join(p.text for p in doc.paragraphs), None)]
    if ext in ("txt", "md"):
        return [(file.getvalue().decode("utf-8", errors="ignore"), None)]
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text, size=180, overlap=35):
    words = re.sub(r"\s+", " ", text).strip().split()
    result, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        part = " ".join(words[start:end])
        if part.strip():
            result.append(part)
        if end == len(words):
            break
        start = end - overlap
    return result


def index_files(files):
    chunks = []
    for file in files:
        number = 0
        for text, page in extract(file):
            for part in chunk_text(text):
                number += 1
                chunks.append(Chunk(part, file.name, number, page))
    return chunks


def lexical_rank(query, chunks):
    docs = [c.text for c in chunks]
    if not docs:
        return []
    word = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                           sublinear_tf=True, max_features=40000)
    try:
        wm = word.fit_transform(docs + [query])
        cm = char.fit_transform(docs + [query])
        scores = (.75 * cosine_similarity(wm[-1], wm[:-1]).ravel()
                  + .25 * cosine_similarity(cm[-1], cm[:-1]).ravel())
    except ValueError:
        return []
    order = np.argsort(scores)[::-1]
    return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


def embed_texts(client, texts, batch_size=64):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def build_semantic_index(chunks, api_key):
    client = OpenAI(api_key=api_key)
    vectors = embed_texts(client, [c.text for c in chunks])
    return vectors


def retrieve(query, chunks, limit, api_key=None, semantic_vectors=None):
    if not chunks:
        return [], "No documents indexed"

    lexical = lexical_rank(query, chunks)
    lexical_ranks = {idx: rank for rank, (idx, _) in enumerate(lexical, 1)}
    semantic_ranks = {}
    semantic_scores = {}

    if api_key and semantic_vectors is not None:
        client = OpenAI(api_key=api_key)
        query_vector = embed_texts(client, [query])[0]
        similarities = semantic_vectors @ query_vector
        order = np.argsort(similarities)[::-1]
        semantic_ranks = {int(idx): rank for rank, idx in enumerate(order, 1)}
        semantic_scores = {int(idx): float(similarities[idx]) for idx in order}

    if semantic_ranks:
        candidates = set(lexical_ranks) | set(semantic_ranks)
        fused = {
            idx: (1 / (RRF_K + lexical_ranks[idx]) if idx in lexical_ranks else 0)
                 + (1 / (RRF_K + semantic_ranks[idx]) if idx in semantic_ranks else 0)
            for idx in candidates
        }
        ranked = sorted(fused, key=fused.get, reverse=True)[:limit]
        method = "Hybrid search (TF-IDF + semantic embeddings, Reciprocal Rank Fusion)"
        return [
            (chunks[idx], fused[idx], {
                "lexical_rank": lexical_ranks.get(idx),
                "semantic_rank": semantic_ranks.get(idx),
                "semantic_similarity": semantic_scores.get(idx),
            })
            for idx in ranked
        ], method

    ranked = lexical[:limit]
    method = "Lexical search (word + character TF-IDF)"
    return [
        (chunks[idx], score, {"lexical_rank": rank, "semantic_rank": None,
                              "semantic_similarity": None})
        for rank, (idx, score) in enumerate(ranked, 1)
    ], method


def generate_answer(question, hits, api_key):
    if not hits:
        return "I couldn't find relevant evidence in the uploaded documents. Try rephrasing the question or adding a source."
    context = "\n\n".join(f"[{chunk.source}]\n{chunk.text}" for chunk, _, _ in hits)
    if not api_key:
        return "No API key configured. Here are the strongest matching passages instead:\n\n" + "\n\n".join(
            f"**{chunk.source}**\n\n> {chunk.text[:650]}" for chunk, _, _ in hits[:3]
        )
    try:
        client = OpenAI(api_key=api_key)
        result = client.chat.completions.create(
            model=GENERATION_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": (
                    "You are WorkplaceIQ, an enterprise knowledge assistant. Answer only from the supplied "
                    "evidence excerpts. Cite every material claim using the exact source label, for example "
                    "[project.pdf, page 2, chunk 1]. Distinguish explicit facts from reasonable inferences. "
                    "If evidence is missing, incomplete, or conflicting, say so. Never invent facts. "
                    "Use concise headings or bullets when helpful."
                )},
                {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
            ],
        )
        return result.choices[0].message.content or "The model returned an empty response."
    except Exception as exc:
        st.warning(f"Answer generation failed ({type(exc).__name__}). Showing retrieved evidence instead.")
        return "Retrieved evidence:\n\n" + "\n\n".join(
            f"**{chunk.source}**\n\n> {chunk.text[:650]}" for chunk, _, _ in hits[:3]
        )


def render_hits(hits, show_details=True):
    """Render current results and legacy two-field results retained in Streamlit sessions."""
    for rank, item in enumerate(hits, 1):
        if len(item) == 3:
            chunk, score, details = item
        elif len(item) == 2:
            # Older app versions stored (chunk, score); Streamlit may retain this
            # session state after a code redeployment.
            chunk, score = item
            details = {
                "lexical_rank": rank,
                "semantic_rank": None,
                "semantic_similarity": None,
            }
        else:
            st.warning("Skipped an evidence result with an unexpected format.")
            continue

        with st.expander(f"#{rank} · {chunk.source} · rank score {score:.4f}"):
            st.write(chunk.text)
            if show_details:
                cols = st.columns(3)
                cols[0].metric("Lexical rank", details.get("lexical_rank") or "—")
                cols[1].metric("Semantic rank", details.get("semantic_rank") or "—")
                semantic_score = details.get("semantic_similarity")
                cols[2].metric(
                    "Semantic similarity",
                    f"{semantic_score:.3f}" if semantic_score is not None else "—",
                )


st.title("WorkplaceIQ")
st.caption("Enterprise document intelligence with hybrid retrieval and inspectable evidence.")
st.write("Upload workplace documents, ask questions across them, and trace answers back to the source passages.")

with st.sidebar:
    st.header("Workspace settings")
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    api_key = st.text_input(
        "OpenAI API key", value=secret_key, type="password",
        help="Powers semantic embeddings and grounded answer generation. Leave blank for lexical-only search."
    )
    top_k = st.slider("Evidence passages", 2, 10, 5)
    st.divider()
    st.caption("Supported formats: PDF, DOCX, TXT, Markdown")
    st.caption("Embedding model: text-embedding-3-small")
    st.caption("Answer model: gpt-4o-mini")
    st.caption("Search scores are ranking signals, not calibrated confidence probabilities.")

key_fingerprint = hashlib.sha256(api_key.encode()).hexdigest() if api_key else None
if st.session_state.get("api_key_fingerprint") != key_fingerprint:
    st.session_state.semantic_vectors = None
    st.session_state.semantic_fingerprint = None
    st.session_state.api_key_fingerprint = key_fingerprint


uploads = st.file_uploader("Upload project documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)

for key, default in {
    "chunks": [], "fingerprint": None, "semantic_vectors": None,
    "semantic_fingerprint": None, "last_hits": [], "answer": "",
    "search_method": "", "inspection": [], "api_key_fingerprint": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if uploads:
    digest = hashlib.sha256()
    for file in uploads:
        digest.update(file.name.encode())
        digest.update(file.getvalue())
    fingerprint = digest.hexdigest()
    if fingerprint != st.session_state.fingerprint:
        with st.spinner("Extracting and indexing documents..."):
            try:
                st.session_state.chunks = index_files(uploads)
                st.session_state.fingerprint = fingerprint
                st.session_state.semantic_vectors = None
                st.session_state.semantic_fingerprint = None
                st.session_state.last_hits = []
                st.session_state.answer = ""
            except Exception as exc:
                st.error(f"Document processing failed: {exc}")
    names = sorted({chunk.filename for chunk in st.session_state.chunks})
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(names))
    col2.metric("Evidence chunks", len(st.session_state.chunks))
    col3.metric("File formats", len({name.rsplit(".", 1)[-1].lower() for name in names}))
    st.caption("Indexed: " + " · ".join(names))
else:
    st.info("Upload one or more documents to begin.")
    st.session_state.chunks = []
    st.session_state.fingerprint = None
    st.session_state.semantic_vectors = None
    st.session_state.semantic_fingerprint = None

ask_tab, inspect_tab, architecture_tab = st.tabs(
    ["Ask your workspace", "Retrieval inspector", "Architecture"]
)

with ask_tab:
    st.subheader("Ask across your documents")
    question = st.text_area(
        "Your question",
        placeholder="What risks, blockers, deadlines, or unresolved decisions are mentioned?",
        height=90,
    )
    if st.button("Search and answer", type="primary",
                 disabled=not st.session_state.chunks or not question.strip()):
        semantic_vectors = None
        if api_key:
            try:
                if st.session_state.semantic_vectors is None:
                    with st.spinner("Creating semantic embeddings for your documents..."):
                        st.session_state.semantic_vectors = build_semantic_index(
                            st.session_state.chunks, api_key
                        )
                        st.session_state.semantic_fingerprint = st.session_state.fingerprint
                semantic_vectors = st.session_state.semantic_vectors
            except Exception as exc:
                st.warning(
                    f"Semantic indexing was unavailable ({type(exc).__name__}). "
                    "Continuing with lexical retrieval."
                )
        hits, method = retrieve(
            question, st.session_state.chunks, top_k, api_key, semantic_vectors
        )
        st.session_state.last_hits = hits
        st.session_state.search_method = method
        st.session_state.answer = generate_answer(question, hits, api_key)

    if st.session_state.answer:
        st.caption("Retrieval mode: " + st.session_state.search_method)
        st.markdown("#### Answer")
        st.markdown(st.session_state.answer)
        st.markdown("#### Evidence used")
        render_hits(st.session_state.last_hits)

with inspect_tab:
    st.subheader("Inspect and compare retrieval")
    query = st.text_input("Test a retrieval query", placeholder="e.g. outstanding invoice amount")
    if st.button("Inspect passages", disabled=not st.session_state.chunks or not query.strip()):
        semantic_vectors = st.session_state.semantic_vectors
        if api_key and semantic_vectors is None:
            try:
                with st.spinner("Creating semantic embeddings..."):
                    semantic_vectors = build_semantic_index(st.session_state.chunks, api_key)
                    st.session_state.semantic_vectors = semantic_vectors
                    st.session_state.semantic_fingerprint = st.session_state.fingerprint
            except Exception as exc:
                st.warning(f"Semantic indexing unavailable ({type(exc).__name__}); using lexical search.")
        hits, method = retrieve(query, st.session_state.chunks, top_k, api_key, semantic_vectors)
        st.session_state.inspection = hits
        st.session_state.inspection_method = method
    if st.session_state.inspection:
        st.caption("Retrieval mode: " + st.session_state.get("inspection_method", ""))
        render_hits(st.session_state.inspection)

with architecture_tab:
    st.subheader("System architecture")
    st.code(
        "PDF / DOCX / TXT / Markdown\n"
        "        ↓\n"
        "Text extraction + page metadata\n"
        "        ↓\n"
        "Overlapping chunks + source identifiers\n"
        "        ↓\n"
        " ┌──────────────────┬─────────────────────┐\n"
        " │ Word/char TF-IDF │ OpenAI text          │\n"
        " │ lexical search   │ embeddings           │\n"
        " └──────────────────┴─────────────────────┘\n"
        "        ↓\n"
        "Reciprocal Rank Fusion (rank-based hybrid)\n"
        "        ↓\n"
        "Top evidence passages\n"
        "        ↓\n"
        "GPT-4o-mini grounded answer + source citations\n"
        "        ↓\n"
        "Evidence inspector",
        language="text",
    )
    st.markdown("""
**Engineering decisions**
- Hybrid retrieval combines exact-term matching with semantic similarity.
- Reciprocal Rank Fusion combines rankings rather than adding incomparable raw scores.
- Document embeddings are generated once per uploaded document set per session and reused for subsequent queries.
- The answer model receives only retrieved passages and is instructed to cite source-level evidence.
- If semantic indexing or generation fails, the app degrades to lexical retrieval or evidence display.

**Current prototype boundaries**
- Session-scoped, in-memory document and embedding storage
- No authentication, multi-tenant isolation, or document-level authorization
- No OCR for scanned PDFs
- No persistent vector database or background ingestion workers
- No calibrated confidence or production retrieval benchmark yet
""")
    st.caption("Independent portfolio project. Not affiliated with or endorsed by American Express.")
