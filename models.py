import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reranker model is cross-encoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# load Groq API
load_dotenv()
groq_api_key = os.environ.get("GROQAPI")
llm_rag = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0)