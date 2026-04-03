# import from other files
from reranking import rerank
from retrieval import hybrid_retrieve
from langchain_core.prompts import ChatPromptTemplate
from models import llm_rag

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analyst assistant.
    Answer the user's question using only the context provided below.
    If the answer is not contained in the context, say "I don't have enough information to answer this question."
    Be precise with numbers, dates, and financial figures.
    
    Context: {context}"""),
    
    ("user", "{question}")
])

chain = prompt_template | llm_rag

def rag(question, corpus, faiss_index, bm25_index, candidate_n=20, top_n=10, top_k=5):
    """
    Orchestrates the full RAG pipeline: retrieval, reranking, and LLM generation.
    Injects the retrieved context into the prompt to answer the user's question.
    """
    retrieved_chunks, retrieved_indices = hybrid_retrieve(question, corpus, faiss_index, bm25_index, top_n=top_n, candidate_n=candidate_n)
    
    reranked_chunks = rerank(question, retrieved_chunks, top_k=top_k)
    
    context = "\n\n".join(reranked_chunks)
    
    response = chain.invoke({"context": context, "question": question})
    

    return response.content