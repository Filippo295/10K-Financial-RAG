# import from other files
from chunking import chunking
from indexing import embedding_indexing
from summarizer import summarize_document

def process_pdf(file):
    """
    Coordinates the full document processing pipeline. 
    It takes an uploaded file, performs chunking, generates embeddings and indexes, 
    and produces a summary of the document.
    """
    pdf_path = file.name

    chunks = chunking(pdf_path)

    corpus, faiss_index, bm25_index = embedding_indexing(chunks)

    summary = summarize_document(corpus)


    return summary, corpus, faiss_index, bm25_index