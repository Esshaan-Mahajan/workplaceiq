# WorkplaceIQ

**An inspectable enterprise document RAG prototype.** Upload project documents, ask questions across them, and inspect the passages retrieved as evidence.

Independent portfolio project. Not affiliated with or endorsed by American Express.

## Features

- PDF, DOCX, TXT, and Markdown ingestion
- Overlapping text chunking
- Hybrid lexical retrieval using word and character TF-IDF
- Optional OpenAI answer generation with source citations
- Evidence-only fallback when no API key is configured
- Retrieval inspector with ranked passages and scores
- Explicit handling of insufficient evidence

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional answer generation

Enter an OpenAI API key in the app sidebar. Without a key, retrieval still works and the app displays the strongest evidence passages.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud and connect GitHub.
3. Create an app using `app.py` as the entry point.
4. Deploy. No secrets are required for evidence-only mode.
5. For generated answers, enter an API key in the sidebar.

## Architecture

```text
PDF / DOCX / TXT / Markdown
          |
          v
    Text extraction
          |
          v
  Overlapping chunking
          |
          v
  Word TF-IDF + character TF-IDF
          |
          v
     Ranked passages
          |
          v
 Optional grounded LLM answer
          |
          v
   Answer + source citations
```

## Important limitations

This is a demonstrable prototype, not a production enterprise platform. It currently has no authentication, multi-tenant isolation, persistent document storage, OCR, calibrated confidence scores, or document-level access-control enforcement. Retrieval scores are ranking signals, not probabilities.

## Next engineering milestones

1. Add semantic embeddings and compare retrieval quality against the lexical baseline.
2. Add an evaluation set with Recall@K, MRR, and citation-groundedness checks.
3. Add document metadata, permissions, and tenant isolation.
4. Add ingestion observability, retries, and structured failure reporting.
5. Add human approval for future agentic actions.
