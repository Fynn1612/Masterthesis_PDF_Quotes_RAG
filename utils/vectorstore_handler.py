import os
import asyncio

from utils.config import GOOGLE_API_KEY, MODEL_OPTIONS
from utils.pdf_handler import get_pdf_text, get_text_chunks

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PROJECT_NAME = "Masterarbeit_RAG_PDFs"

GOOGLE_DRIVE_BASE = r"G:\Meine Ablage"

# Mapping each model provider to its corresponding persistent directory for storing vectorstore data
PERSIST_DIR = {
  key.lower(): os.path.join(
    GOOGLE_DRIVE_BASE, PROJECT_NAME, "Vectorsores", key.lower(), "chroma_db"
    
  )
  for key in MODEL_OPTIONS.keys()
}

def get_embeddings(model_provider):
    """
    Returns the appropriate embedding model based on the selected model provider.

    - For 'groq', returns a HuggingFace MiniLM embedding model.
    - For 'gemini', returns Google's Generative AI embedding model.

    Raises:
      ValueError: If the given provider is not supported.
    """
    if model_provider == "groq":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2"
        )
    elif model_provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=GOOGLE_API_KEY
        )
    else:
        raise ValueError("Unsupported Model Provider")

def get_files_from_folder():
    """
    Simulates file upload by reading all PDF files from a specified Google Drive folder.
    
    This function:
    - Reads all PDF files from the specified Google Drive folder.
    - Simulates the file upload process by returning file-like objects.

    Returns:
    list: A list of file-like objects representing the uploaded PDF files.
    """
    
    folder_path = os.path.join(GOOGLE_DRIVE_BASE, PROJECT_NAME, "PDFs")
    uploaded_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            f = open(file_path, "rb")
            #f.name = filename  # Simulate the 'name' attribute of uploaded files
            uploaded_files.append(f) 
                           
    return uploaded_files
    

def get_or_create_vectorstore(uploaded_files, model_provider):
    """
    Loads an existing Chroma vectorstore from disk if it exists, or creates a new one from uploaded PDFs.

    This function:
    - Extracts raw text and metadata from uploaded PDFs.
    - Splits the text into chunks suitable for embedding while keeping metadata.
    - Loads or creates a vectorstore for the given model provider.
    - Appends to existing vectorstore if already present.

    Args:
      uploaded_files (list): List of uploaded PDF files.
      model_provider (str): Lowercase name of the selected model provider ('groq' or 'gemini').

    Returns:
      Chroma: A Chroma vectorstore containing embedded PDF text chunks.
    """
    # Ensure an asyncio event loop is available for Gemini
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Extract raw text from the uploaded PDF files with metadata
    raw_text_with_metadata = get_pdf_text(uploaded_files)

    # Chunk the raw text for embedding (e.g., 5000 characters with overlap) with metadata
    chunks = get_text_chunks(raw_text_with_metadata)

    # Load the appropriate embedding model
    embedding = get_embeddings(model_provider)

    # Define directory path to store or retrieve Chroma DB
    persist_path = PERSIST_DIR[model_provider]
    os.makedirs(persist_path, exist_ok=True)

    # If the vectorstore directory exists and is not empty, load and append new chunks
    if os.path.exists(persist_path) and os.listdir(persist_path):
        vectorstore = Chroma(
            persist_directory=persist_path, embedding_function=embedding
        )
               
        vectorstore.add_texts(
            texts=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks]
        )
       
        vectorstore.persist()
    else:
        # Otherwise, create a new vectorstore from the chunks
        vectorstore = Chroma.from_texts(
            texts=[c["text"] for c in chunks],
            embedding=embedding,
            metadatas=[c["metadata"] for c in chunks],
            persist_directory=persist_path
        )
        vectorstore.persist()

    return vectorstore
