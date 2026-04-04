# 10-K Financial Analyst

A RAG pipeline that lets you upload any SEC 10-K filing and ask questions about it. The system chunks and indexes the document on upload, generates an executive summary via Map-Reduce, then answers financial questions using hybrid retrieval, cross-encoder reranking, and Llama 3.3 70B. Deployed on HuggingFace Spaces via FastAPI and Docker.

**Live Demo:** https://huggingface.co/spaces/Filippo295/10K-Financial-RAG

---

## The Problem

10-K filings are hundreds of pages of dense financial and legal text. Extracting specific figures or understanding a company's strategy requires either reading the entire document or knowing exactly where to look and this is very time consuming for investors whose time is literally money. This project wraps the full document into a RAG pipeline so you can query it conversationally and get precise answers without reading hundreds of pages.

---

## Architecture

The pipeline is split into two phases triggered at different times.

**At PDF upload:**
1. The document is extracted with PyMuPDF and split into overlapping chunks (size=2000, overlap=400) via `RecursiveCharacterTextSplitter`
2. Chunks are embedded with `all-MiniLM-L6-v2` and indexed into a FAISS index (cosine similarity) and a BM25 index (keyword matching) in parallel
3. A Map-Reduce summarizer generates an executive summary of the company: the first 100 chunks (which more or less coincide with the company overview) are grouped into batches of 10, each batch is summarized independently (Map), then all mini-summaries are synthesized into a structured report with Business Overview, Revenue Drivers, Risk Factors, and Strategic Outlook (Reduce)

**At each query:**
1. FAISS semantic search and BM25 keyword search each retrieve 20 candidate chunks independently. Reciprocal Rank Fusion (RRF) merges the two ranked lists and selects the top 10 chunks overall.
2. A `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder scores each of the 10 (query, chunk) pairs and selects the top 5
3. The reranked chunks are injected as context into a LangChain prompt and passed to Llama 3.3 70B via Groq API

The interface is built with Gradio. Each user session gets isolated state via `gr.State()`, so multiple users uploading different PDFs do not overwrite each other's index.

---

## Retrieval Evaluation

Retrieval quality was measured with Mean Reciprocal Rank (MRR) on 10 simulated financial questions against Tesla's 10-K filing. Each question has an expected answer containing specific figures; MRR measures how high the correct chunk ranks in the reranked results.

**MRR: 0.75**

---

## Key Design Decisions

- Hybrid retrieval (FAISS + BM25 + RRF) was chosen over either method alone: embeddings capture semantic meaning but struggle with exact numerical figures and specific financial terms, while BM25 keyword matching finds chunks containing the exact query terms but misses conceptually related content
- Cross-encoder reranking was added as a second-stage filter because cross-encoders read the query and each chunk together, making them significantly more accurate at judging relevance than the bi-encoders used in the retrieval stage. The tradeoff is that cross-encoders are too slow to run on the full corpus, which is why they only rescore the top 10 candidates from retrieval
- A Map-Reduce summarizer runs automatically at upload to give the user an immediate overview of the company before asking any questions. Passing the full document to the LLM directly proved to be too expensive in tokens, so only the first 100 chunks are used, which correspond roughly to the business overview section of a 10-K where company description, strategy, and risk factors are concentrated


---

## Limitations and Possible Improvements

- The system is synchronous, meaning requests are queued and a slow query from one user blocks all others. This could be addressed with async request handling
- Latency has two sources. At upload, the bottleneck is the Map-Reduce summarizer which makes 11 LLM calls (10 map + 1 reduce); chunking and embedding add some overhead especially on longer documents but are still significantly faster, while indexing is negligible. At query time, retrieval and reranking are fast and the bottleneck is the single LLM call where the retrieved context is injected into the prompt. Implementing streaming would reduce the perceived latency by showing the answer word by word as it is generated, similar to how LLM chatbots work
- The MRR evaluation uses only 10 questions on a single document (Tesla 10-K), which is a narrow benchmark. Evaluating across multiple filings and question types would give a more reliable picture of retrieval quality

---

## Tech Stack

**RAG & Retrieval:** `faiss-cpu` (semantic search), `rank-bm25` (keyword search), Reciprocal Rank Fusion, `sentence-transformers` with `all-MiniLM-L6-v2` (embeddings), `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranking)

**LLM & Orchestration:** Llama 3.3 70B via Groq API, `langchain`, `langchain-groq`

**Document Processing:** `pymupdf`, `langchain-text-splitters`

**UI & Serving:** Gradio (`gr.Blocks`), FastAPI, Uvicorn

**Deployment:** Docker, HuggingFace Spaces (port 7860)
