import hashlib
import re
from dataclasses import dataclass

import numpy as np
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="WorkplaceIQ", page_icon="🔎", layout="wide")

@dataclass
class Chunk:
    text: str
    filename: str
    number: int
    page: int | None = None

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
        result.append(" ".join(words[start:end]))
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

def retrieve(query, chunks, limit):
    docs = [c.text for c in chunks]
    word = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, max_features=40000)
    wm = word.fit_transform(docs + [query])
    cm = char.fit_transform(docs + [query])
    scores = .75 * cosine_similarity(wm[-1], wm[:-1]).ravel() + .25 * cosine_similarity(cm[-1], cm[:-1]).ravel()
    order = np.argsort(scores)[::-1][:limit]
    return [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]

def generate_answer(question, hits, api_key):
    if not hits:
        return "I couldn't find relevant evidence in the uploaded documents. Try rephrasing the question or adding a source."
    context = "\n\n".join(
        f"[{c.filename}" + (f", page {c.page}" if c.page else "") + f", chunk {c.number}]\n{c.text}"
        for c, _ in hits
    )
    if not api_key:
        return "No LLM key configured. Here are the strongest matching passages instead:\n\n" + "\n\n".join(
            f"**{c.filename}" + (f", page {c.page}" if c.page else "") + f" (chunk {c.number})**\n\n> {c.text[:650]}"
            for c, _ in hits[:3]
        )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role":"system","content":"You are WorkplaceIQ, an enterprise knowledge assistant. Answer only from the supplied excerpts. Cite each material claim with the exact source label, such as [file.pdf, page 2, chunk 1]. If evidence is missing or conflicting, say so. Never invent facts."},
                {"role":"user","content":f"Question: {question}\n\nEvidence:\n{context}"}
            ]
        )
        return result.choices[0].message.content
    except Exception as exc:
        st.warning(f"Answer generation failed ({type(exc).__name__}). Showing retrieved evidence instead.")
        return "Retrieved evidence:\n\n" + "\n\n".join(f"**{c.filename}**\n\n> {c.text[:650]}" for c, _ in hits[:3])

st.title("WorkplaceIQ")
st.caption("Enterprise knowledge retrieval with evidence you can inspect.")
st.write("Upload workplace documents, ask questions across them, and inspect the passages retrieved as evidence.")

with st.sidebar:
    st.header("Workspace settings")
    api_key = st.text_input("OpenAI API key (optional)", type="password", help="Used for grounded answer generation. Retrieval works without it.")
    top_k = st.slider("Evidence passages", 2, 10, 5)
    st.divider()
    st.caption("Supported: PDF, DOCX, TXT, Markdown")
    st.caption("Prototype: lexical retrieval using word and character TF-IDF. Scores rank passages; they are not calibrated confidence probabilities.")

uploads = st.file_uploader("Upload project documents", type=["pdf","docx","txt","md"], accept_multiple_files=True)
if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.fingerprint = None
    st.session_state.last_hits = []
    st.session_state.answer = ""

if uploads:
    digest = hashlib.sha256()
    for f in uploads:
        digest.update(f.name.encode())
        digest.update(f.getvalue())
    fingerprint = digest.hexdigest()
    if fingerprint != st.session_state.fingerprint:
        with st.spinner("Extracting and indexing documents..."):
            try:
                st.session_state.chunks = index_files(uploads)
                st.session_state.fingerprint = fingerprint
                st.session_state.last_hits = []
                st.session_state.answer = ""
            except Exception as exc:
                st.error(f"Document processing failed: {exc}")
    names = sorted({c.filename for c in st.session_state.chunks})
    a,b,c = st.columns(3)
    a.metric("Documents", len(names))
    b.metric("Evidence chunks", len(st.session_state.chunks))
    c.metric("File formats", len({n.rsplit(".",1)[-1].lower() for n in names}))
    st.caption("Indexed: " + " · ".join(names))
else:
    st.info("Upload one or more documents to begin.")
    st.session_state.chunks = []
    st.session_state.fingerprint = None

ask_tab, inspect_tab, architecture_tab = st.tabs(["Ask your workspace", "Retrieval inspector", "Architecture"])
with ask_tab:
    st.subheader("Ask across your documents")
    question = st.text_area("Your question", placeholder="What risks, blockers, deadlines, or unresolved decisions are mentioned?", height=90)
    if st.button("Search and answer", type="primary", disabled=not st.session_state.chunks or not question.strip()):
        hits = retrieve(question, st.session_state.chunks, top_k)
        st.session_state.last_hits = hits
        st.session_state.answer = generate_answer(question, hits, api_key)
    if st.session_state.answer:
        st.markdown("#### Answer")
        st.markdown(st.session_state.answer)
        st.markdown("#### Retrieved sources")
        for chunk, score in st.session_state.last_hits:
            location = f" · page {chunk.page}" if chunk.page else ""
            with st.expander(f"{chunk.filename}{location} · chunk {chunk.number} · score {score:.3f}"):
                st.write(chunk.text)

with inspect_tab:
    st.subheader("Inspect retrieval")
    query = st.text_input("Test a retrieval query", placeholder="e.g. outstanding invoice amount")
    if st.button("Inspect passages", disabled=not st.session_state.chunks or not query.strip()):
        st.session_state.inspection = retrieve(query, st.session_state.chunks, top_k)
    for rank, (chunk, score) in enumerate(st.session_state.get("inspection", []), 1):
        st.markdown(f"**#{rank} · {chunk.filename}** · score: {score:.4f}")
        st.write(chunk.text)
        st.divider()

with architecture_tab:
    st.subheader("System flow")
    st.code("Documents → text extraction → overlapping chunks → word + character TF-IDF retrieval → ranked evidence → optional grounded LLM answer → citations + inspection", language="text")
    st.markdown("""
**Reliability decisions**
- The model is instructed to answer only from retrieved excerpts.
- Material claims must cite their source document and passage.
- Missing or conflicting evidence should be stated explicitly.
- Without an API key, the app returns source passages instead of fabricating a synthesis.
- Retrieval scores are ranking signals, not probabilities.

**Prototype boundaries**
- No authentication or multi-tenant isolation
- No persistent document storage
- No OCR for scanned PDFs
- No document-level permission enforcement
""")
    st.caption("Independent portfolio project. Not affiliated with or endorsed by American Express.")
