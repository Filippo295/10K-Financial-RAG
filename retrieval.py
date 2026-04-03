import faiss
import numpy as np
# import from other files
from models import embedding_model

def hybrid_retrieve(query, corpus, faiss_index, bm25_index, top_n=10, candidate_n=20, rrf_n=60): 
    """
    Retrieves the most relevant chunks for a query using Hybrid Retrieval.
    Combines semantic search (FAISS) and keyword search (BM25),
    then fuses the rankings with Reciprocal Rank Fusion.
    """
    # SEMANTIC SEARCH:
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    
    query_embedding = query_embedding.reshape(1, -1) 
    faiss.normalize_L2(query_embedding)
    
    scores, indices = faiss_index.search(query_embedding, candidate_n)
    semantic_results = indices[0].tolist()

    
    # BM25 SEARCH:
    tokenized_query = query.lower().split()
    
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_results = np.argsort(bm25_scores)[::-1][:candidate_n].tolist()

    
    # RRF:
    rrf_scores = {}
    
    for rank, idx in enumerate(semantic_results):
        if idx not in rrf_scores:
            rrf_scores[idx] = 0.0
        rrf_scores[idx] += 1 / (rrf_n + rank + 1)
    
    for rank, idx in enumerate(bm25_results):
        if idx not in rrf_scores:
            rrf_scores[idx] = 0.0
        rrf_scores[idx] += 1 / (rrf_n + rank + 1)
    
    top_n_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_n]
    
    top_n_chunks = [corpus[idx] for idx in top_n_indices]

    
    return top_n_chunks, top_n_indices