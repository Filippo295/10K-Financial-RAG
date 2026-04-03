# evaluation script: run this standalone to measure retrieval quality
# requires data/tsla10k.pdf in the project folder
# the MRR result is reported in the README
import numpy as np
# import from other files
from chunking import chunking
from indexing import embedding_indexing
from retrieval import hybrid_retrieve
from reranking import rerank

# 10 simulated questions
eval_dataset = [
    {"question": "What were Tesla's total revenues in 2024?",
     "keywords": ["97,690", "total revenues"]},
    {"question": "What was Tesla's net income in 2024?",
     "keywords": ["7,153", "net income"]},
    {"question": "What was Tesla's gross profit in 2024?",
     "keywords": ["17,450", "gross profit"]},
    {"question": "How much did Tesla spend on research and development in 2024?",
     "keywords": ["4,540", "research and development"]},
    {"question": "What was Tesla's total inventory as of December 31 2024?",
     "keywords": ["12,017", "total"]},
    {"question": "What was Tesla's total property plant and equipment net in 2024?",
     "keywords": ["35,836", "property, plant and equipment, net"]},
    {"question": "What was Tesla's total debt and finance leases as of December 31 2024?",
     "keywords": ["5,757", "total debt and finance leases"]},
    {"question": "What was Tesla's provision for income taxes in 2024?",
     "keywords": ["1,837", "provision for"]},
    {"question": "What were Tesla's total accrued liabilities in 2024?",
     "keywords": ["10,723", "total"]},
    {"question": "What was Tesla's income before income taxes in 2024?",
     "keywords": ["8,990", "income before income taxes"]},
]

# create corpus, faiss_index and bm25_index globally so compute_mrr can access them
# normally they only exist inside the Gradio pipeline, passed between functions and never exposed globally
pdf_path = "data/tsla10k.pdf"
chunks = chunking(pdf_path)
corpus, faiss_index, bm25_index = embedding_indexing(chunks)

# MRR computation
def compute_mrr(eval_dataset, corpus, faiss_index, bm25_index, candidate_n=20, top_n=10, top_k=5):
    """
    Evaluates retrieval quality using Mean Reciprocal Rank (MRR).
    For each simulated question retrieves and reranks chunks,
    then checks at what position the correct chunk appears.
    MRR is the average of 1/rank across all questions. 
    Score is between 0 and 1, higher is better.
    """
    reciprocal_ranks = []

    for item in eval_dataset:

        retrieved_chunks, _ = hybrid_retrieve(item["question"], corpus, faiss_index, bm25_index, top_n=top_n, candidate_n=candidate_n)

        reranked_chunks = rerank(item["question"], retrieved_chunks, top_k=top_k)

        rank_found = None
        for rank, chunk in enumerate(reranked_chunks):
            if all(kw.lower() in chunk.lower() for kw in item["keywords"]):
                rank_found = rank + 1
                break

        if rank_found is None:
            reciprocal_ranks.append(0.0)
            print("NOT FOUND", item["question"])
        else:
            reciprocal_ranks.append(1 / rank_found)
            print("Rank", rank_found, "RR", round(1 / rank_found, 2), item["question"])

    mrr = round(np.mean(reciprocal_ranks), 4)
    print("\nMRR: " + str(mrr))

compute_mrr(eval_dataset, corpus, faiss_index, bm25_index)