import uvicorn
from fastapi import FastAPI
import gradio as gr
from gradioui import demo

# create the FastAPI application
app = FastAPI()

# health check
@app.get("/health")
def health():
    return {"status": "ok"}

# mount the Gradio app inside FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

# start the server with uvicorn, 7860 is port for HuggingFace Spaces
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)