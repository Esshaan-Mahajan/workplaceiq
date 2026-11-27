# WorkplaceIQ

**Hybrid enterprise document intelligence with inspectable evidence.**

WorkplaceIQ lets users upload project and workplace documents, ask natural-language questions across them, and inspect the source passages behind each answer. It combines lexical retrieval with semantic embeddings and uses rank fusion to produce a hybrid evidence set for grounded answer generation.

**Live demo:** https://workplaceiq-2jdw7zrl9eavp5hfmgjfev.streamlit.app/

Independent portfolio project. Not affiliated with or endorsed by American Express.

## What it does

- Ingests PDF, DOCX, TXT, and Markdown documents.
- Extracts text and preserves source filename and PDF page metadata.
- Splits documents into overlapping chunks with stable source labels.
- Runs lexical retrieval with word and character TF-IDF.
- Runs semantic retrieval using OpenAI text-embedding-3-small.
- Combines lexical and semantic rankings using Reciprocal Rank Fusion (RRF).
- Generates grounded answers with gpt-4o-mini, with citations to source passages.
- Provides a retrieval inspector that exposes lexical rank, semantic rank, and semantic similarity.
- Falls back to lexical-only retrieval and evidence display when an API key is unavailable or semantic indexing fails.

## Architecture

```text
PDF / DOCX / TXT / Markdown
          |
          v
Text extraction + source metadata
          |
          v
Overlapping chunking
          |
          +---------------------------+
          |                           |
          v                           v
Word + character TF-IDF       OpenAI embeddings
Lexical retrieval             Semantic retrieval
          |                           |
          +-------------+-------------+
                        |
                        v
          Reciprocal Rank Fusion (RRF)
                        |
                        v
                Top evidence chunks
                        |
                        v
           GPT-4o-mini grounded answer
                        |
                        v
           Source citations + inspector
```

### Why Reciprocal Rank Fusion?

Lexical similarity and embedding cosine similarity have different score distributions. Rather than adding raw scores directly, WorkplaceIQ combines their ranked result lists using RRF:

RRF(d) = Σ 1 / (k + rank(d))

This provides a simple rank-based hybrid baseline without assuming that the underlying scores are calibrated or directly comparable.

## Run locally

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Configure the OpenAI API key

The app can run without a key in lexical-only mode. To enable semantic embeddings and generated answers, provide an OpenAI API key.

For local development, enter the key in the app sidebar. Alternatively, create .streamlit/secrets.toml:

```toml
OPENAI_API_KEY = "your-api-key"
```

For Streamlit Community Cloud, add OPENAI_API_KEY under the app's **Settings → Secrets**. Never commit secrets.toml or paste API keys into source code.

The embedding model is text-embedding-3-small; the answer model is gpt-4o-mini. API usage may incur charges.

## Deployment

1. Connect this GitHub repository to Streamlit Community Cloud.
2. Set app.py as the application entry point.
3. Use Python 3.12.
4. Add OPENAI_API_KEY to Streamlit app secrets to enable semantic retrieval and answer generation.
5. Deploy. Without a key, the app remains usable in lexical-only mode.

## Evaluation approach

The repository includes lightweight retrieval metrics (Recall@K and Mean Reciprocal Rank) to support a labeled evaluation set. A meaningful comparison requires query-to-relevant-chunk labels from representative documents. No retrieval-quality improvement is claimed until that evaluation is run on a repeatable dataset.

Recommended comparison:
- Lexical baseline (word + character TF-IDF)
- Semantic retrieval
- Hybrid retrieval with RRF
- Hybrid retrieval plus a reranker (future experiment)

## Current limitations

This is a portfolio prototype, not a production enterprise platform:

- Documents and embeddings are held in Streamlit session memory; they are not durably stored.
- No authentication, tenant isolation, or document-level authorization is implemented.
- No OCR for scanned PDFs.
- No background ingestion workers, persistent vector database, or production observability.
- No calibrated confidence scores.
- LLM answers are grounded by prompting and source display, but are not formally guaranteed to be factually correct. Users should inspect cited evidence.

## Next steps

1. Add a representative labeled evaluation corpus and report Recall@K / MRR.
2. Add a cross-encoder reranker and measure whether it improves retrieval.
3. Add document-level permissions and tenant isolation before handling sensitive enterprise data.
4. Add ingestion status, structured error reporting, and persistent storage.
5. Evaluate citation correctness and unsupported-claim rates.
