from langchain_core.prompts import ChatPromptTemplate
# import from other files
from models import llm_rag

def summarize_document(corpus, group_size=10):
    """
    Summarizes a financial 10-K document using a Map-Reduce strategy. 
    It first generates concise summaries for individual blocks of text (Map phase) 
    and then synthesizes them into a final structured executive report (Reduce phase).
    
    Map phase takes the first 100 chunks of the document (they more or less correspond
    to the business overview of the 10-K), groups them in batches of 10 and summarizes 
    each group, this way we save tokens and make less API calls.
    The Reduce phase makes a summary of the summaries. 
    """
    groups = [corpus[i:i+group_size] for i in range(0, min(100, len(corpus)), group_size)]

    # MAP PHASE
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst reading a section of a 10-K annual report.
        Summarize the key information in the text below in maximum 150 words.
        Focus on: business description, products and services, strategy, revenue drivers, and risk factors.
        Ignore: legal boilerplate, administrative details, and stock market information.
        If the section contains no relevant business or financial information, respond with exactly: NO RELEVANT CONTENT"""),
        ("human", "{text}")
    ])
    map_chain = map_prompt | llm_rag

    mini_summaries = []
    for i, group in enumerate(groups):
        group_text = "\n\n".join(group)
        response = map_chain.invoke({"text": group_text})
        mini_summary = response.content
        if "NO RELEVANT CONTENT" not in mini_summary:
            mini_summaries.append(mini_summary)

    # REDUCE PHASE
    all_mini_summaries = "\n\n".join(mini_summaries)

    reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior financial analyst writing an executive summary of a SEC 10-K annual report.
    Based on the section summaries below, write a structured report with exactly these sections:

    COMPANY NAME
    
    **Business Overview**
    What the company does, its main products and services, key markets and business segments.

    **Revenue Drivers and Business Segments**
    Key sources of revenue, main business segments and what drives growth in each.
    
    **Key Risk Factors**
    The most material operational and financial risks. Ignore generic legal risks and boilerplate.
    
    **Strategic Outlook**
    Growth initiatives, investments, and management priorities for the future.

    Rules:
    - Start with the full company name and stock ticker on the first line, before any section.
    - Each section must be 3-5 sentences
    - Be precise with numbers and dates
    - Ignore legal boilerplate, footnotes, and administrative details"""),
    ("human", "{text}")
    ])
    reduce_chain = reduce_prompt | llm_rag
        
    final_summary = reduce_chain.invoke({"text": all_mini_summaries})


    return final_summary.content