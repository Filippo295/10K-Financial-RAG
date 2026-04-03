import faiss
from rank_bm25 import BM25Okapi
# import from other files
from models import embedding_model

def embedding_indexing(chunks):
    """
    Embeds the chunks and creates a FAISS and a BM25 index used for Hybrid Retrieval.
    """
    # EMBEDDING
    corpus = [chunk.page_content for chunk in chunks]
    
    embeddings = embedding_model.encode(
        corpus,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # FAISS INDEX
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    # BM25 INDEX
    tokenized_corpus = [chunk.lower().split() for chunk in corpus]
    bm25_index = BM25Okapi(tokenized_corpus)


    return corpus, faiss_index, bm25_index
