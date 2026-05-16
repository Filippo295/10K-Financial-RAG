import gradio as gr
from processpdf import process_pdf
# import from other files
from rag import rag 

# Gradio UI: chat interface with PDF upload
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    # gr.State() creates an isolated memory for every user
    # each user gets their own corpus, faiss_index and bm25_index based on their PDF
    # so multiple users uploading different PDFs don't overwrite each other's PDF
    corpus_state = gr.State()
    faiss_state = gr.State()
    bm25_state = gr.State()
    
    gr.Markdown("# 10-K FINANCIAL ANALYST")
    gr.Markdown("Upload a company's 10-K filing and ask questions about it!")

    with gr.Row():
        # USER INPUTS
        with gr.Column():
            in_text = gr.Textbox(label="Ask a question", placeholder="e.g. What were total revenues in 2024?")
            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            with gr.Row():
                run_btn = gr.Button("Ask", variant="primary")
        # RAG OUTPUTS
        with gr.Column():
            out_text = gr.Textbox(label="Answer", lines=12)
            out_summary = gr.Textbox(label="Document Summary", lines=12)

    pdf_upload.change(
        fn=process_pdf, 
        inputs=pdf_upload, 
        outputs=[out_summary, corpus_state, faiss_state, bm25_state]
    )

    run_btn.click(
        fn=rag, 
        inputs=[in_text, corpus_state, faiss_state, bm25_state], 
        outputs=[out_text]
    )
