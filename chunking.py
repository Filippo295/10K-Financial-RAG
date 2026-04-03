import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking(pdf_path, chunk_size=2000, chunk_overlap=400):
    """
    Extracts text from a PDF and splits it into overlapping chunks.
    """
    doc = fitz.open(pdf_path)
    
    full_document = "\n".join([page.get_text() for page in doc])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.create_documents([full_document])


    return chunks