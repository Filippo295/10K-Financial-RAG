# import from other files
from models import reranker

def rerank(query, chunks, top_k=5):
    """
    Reranks a list of candidate chunks based on their relevance to the query. 
    Uses a Cross-Encoder model to give a score to pairs of (query, chunk).
    """
    pairs = [(query, chunk) for chunk in chunks]
    
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    
    top_chunks = [chunk for chunk, score in ranked[:top_k]]


    return top_chunks