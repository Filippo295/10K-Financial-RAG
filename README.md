# 10-K Financial Analyst

A RAG pipeline that lets you upload any SEC 10-K filing and ask questions about it. The system chunks and indexes the document on upload, generates an executive summary via Map-Reduce, then answers financial questions using hybrid retrieval, cross-encoder reranking, and Llama 3.3 70B. Deployed on HuggingFace Spaces with Docker.

Live demo: https://github.com/Filippo295/10K-Financial-RAG/tree/main

---

## The Problem

10-K filings are hundreds of pages of dense financial and legal text. Extracting specific figures or understanding a company's strategy requires either reading the entire document or knowing exactly where to look. This project wraps the full document into a RAG pipeline so you can query it conversationally and get precise, grounded answers.

---

## Architecture

The pipeline is split into two phases triggered at different times.

**At PDF upload:**
1. The document is extracted with PyMuPDF and split into overlapping chunks (size=2000, overlap=400) via `RecursiveCharacterTextSplitter`
2. Chunks are embedded with `all-MiniLM-L6-v2` and indexed into a FAISS flat inner-product index (L2-normalized for cosine similarity) and a BM25 index in parallel
3. A Map-Reduce summarizer generates an executive summary: the first 100 chunks are grouped into batches of 10, each batch is summarized independently (Map), then all mini-summaries are synthesized into a structured report with Business Overview, Revenue Drivers, Risk Factors, and Strategic Outlook (Reduce)

**At each query:**
1. Hybrid retrieval fuses FAISS semantic search and BM25 keyword search via Reciprocal Rank Fusion (RRF), producing a candidate set of 20 chunks
2. A `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder reranks the candidates and selects the top 5
3. The reranked chunks are injected as context into a LangChain prompt and passed to Llama 3.3 70B via Groq API

Each user session gets isolated state via `gr.State()`, so multiple users uploading different PDFs do not overwrite each other.

---

## Retrieval Evaluation

The pipeline was evaluated on 10 financial questions against Tesla's 2024 10-K using Mean Reciprocal Rank (MRR). Each question was matched against the retrieved and reranked chunks by checking for the presence of the expected numerical figure and label.

**MRR: 0.90**

---

## Key Design Decisions

- Hybrid retrieval (FAISS + BM25 + RRF) was chosen over pure semantic search because financial documents contain precise numerical figures that keyword matching handles better than embeddings
- Cross-encoder reranking was added after hybrid retrieval because bi-encoder similarity scores are not directly comparable across the two retrieval methods; the cross-encoder rescores all candidates jointly against the query
- Map-Reduce summarization was used instead of passing the full document to the LLM to stay within context limits and reduce token cost; only the first 100 chunks are used as they correspond roughly to the business overview section of a 10-K

---

## Limitations and Possible Improvements

- Multiple users are queued rather than served concurrently. The system is not async, so a slow upload from one user blocks others. This could be fixed with async request handling
- Latency per query is dominated by the LLM call when the retrieved context is injected into the prompt. Implementing streaming would make the perceived latency significantly lower, similar to how ChatGPT renders words progressively
- The MRR evaluation uses only 10 questions on a single document (Tesla 2024 10-K), which is a narrow benchmark. Evaluating across multiple filings and question types would give a more reliable picture of retrieval quality

---

## Tech Stack

**RAG & Retrieval:** `faiss-cpu` (semantic search), `rank-bm25` (keyword search), Reciprocal Rank Fusion, `sentence-transformers` with `all-MiniLM-L6-v2` (embeddings), `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranking)

**LLM & Orchestration:** Llama 3.3 70B via Groq API, `langchain`, `langchain-groq`, `langchain-core`

**Document Processing:** `pymupdf`, `langchain-text-splitters`

**UI & Serving:** Gradio (`gr.Blocks`), FastAPI, Uvicorn

**Deployment:** Docker, HuggingFace Spaces (port 7860)
