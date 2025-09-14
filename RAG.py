import json
import os
from datetime import datetime
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

from utils.chat_history_handling import format_chat_history
from utils.config import GOOGLE_API_KEY
from utils.hash_handling import compute_md5, load_hashes, save_hashes

# --------------------
# CONFIG
# --------------------
PDF_FOLDER = "G:/Meine Ablage/Masterarbeit_RAG_PDFs/PDFs"
CHROMA_PATH = r"G:\Meine Ablage\Masterarbeit_RAG_PDFs\Vectorstores\huggingface\chroma_db"
CHAT_HISTORY_DIR = "chat_histories" 


llm_model_name = "gemini-2.5-flash"

@st.cache_resource
def get_embeddings():
    """Get embeddings for text documents.

    Returns:
        HuggingFaceEmbeddings: The embeddings model.
    """
    embeddings = HuggingFaceEmbeddings(
        #model_name="BAAI/bge-base-en-v1.5"
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

def load_documents():
    """Load PDF documents from the specified folder.
    Each Document/Page has a page_content and metadata including source and page number.

    Returns:
        list[Document]: A list of loaded PDF documents. Each entry of the list is one page of one document.
    """
    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    documents = loader.load()
    return documents

def split_documents(documents: list[Document]):
    """Split documents/pages into smaller chunks for processing.
    Each chunks has a page_content and metadata including source, page number and a page_label if 
    one page is split into several pages.

    Args:
        documents (list[Document]): A list of documents to split.

    Returns:
        list[Document]: A list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def calculate_chunk_ids(chunks):
    """
    Calculate unique IDs for each document chunk based on source, page number, and chunk index.
    This will create IDs like "data/monopoly.pdf:6:2"
    Source : Page Number : Chunk Index
    
    Args:
        chunks (list[Document]): A list of document chunks.
    
    Returns:
        list[Document]: The list of document chunks with updated metadata including unique IDs.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(chunks: list[Document]):
    """Add document chunks to the Chroma vector store.
    

    Args:
        chunks (list[Document]): A list of document chunks to add.
    """
    # Load the existing Chroma DB.
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = get_embeddings(),
    )
    
    #calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # load stored hashes
    stored_hashes = load_hashes()
    new_hashes = {}
    
    new_chunks = []
    for chunk in chunks_with_ids:
        source = chunk.metadata.get("source")
        filepath = os.path.join(PDF_FOLDER, os.path.basename(source))
        file_hash = compute_md5(filepath)
        new_hashes[os.path.basename(source)] = file_hash

        if stored_hashes.get(os.path.basename(source)) != file_hash:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
    else:
        print("✅ No new documents to add")
    
    # Save updated hashes
    save_hashes(new_hashes)

def format_docs(docs):
    """Format documents for context in the prompt.
    Each document is formatted with its source and page number."""
    
    return "\n\n".join(
        f"Quelle: {doc.metadata.get('source', 'unbekannt')} (Seite {doc.metadata.get('page', '?')})\n"
        f"{doc.page_content}"
        for doc in docs
    )

@st.cache_resource
def _build_rag_chain(llm_model_name=llm_model_name):
    """
    Builds the RAG (Retrieval-Augmented Generation) chain for answering questions based on PDF content.
    If there are new/changed documents, splits and embeds them, otherwise loads the existing vectorstore.
    Returns the complete RAG chain.
    """
    # Get embedding model (ensures event loop)
    embedding = get_embeddings()
    # Load the vector store (Chroma)
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
    )
    # Load the LLM (Google Gemini)
    llm = ChatGoogleGenerativeAI(model=llm_model_name, api_key=GOOGLE_API_KEY)

    # Multi-query prompt: helps retrieve more relevant chunks by rephrasing the question
    multi_query_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are a scientific AI assistant. Generate 5 different versions of the user question "
            "to improve document retrieval from a vector database.\n"
            "Original question: {question}"
        ),
    )
    # Multi-query retriever: retrieves relevant chunks using multiple question variants
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(),
        llm=llm,
        prompt=multi_query_prompt,
    )

    print("[✓] Retriever with multi-query setup complete.")

    # Context prompt: instructs the LLM to answer using only the provided context and to cite sources
    context_prompt = ChatPromptTemplate.from_template(
        "You are an expirenced Expert in Machine Learning and Deep Drawing. "
        "Answer the question using ONLY the context below."
        "Here is the previous chat history:\n{chat_history}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Provide the source(filename) and the page number of the chunks you used."
    )
    # Build the RAG chain: retrieves context, formats prompt, gets LLM answer, parses output
    rag_chain = (
        RunnableMap(
            {
                "context": retriever | format_docs,
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
        )
        | context_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def main():
    # Set up Streamlit page
    st.set_page_config(page_title="Masterthesis Literature Chatbot", layout="centered")
    st.title("Masterthesis Literature Chatbot")
    st.caption("Chat with multiple PDFs :books:")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar: optinal export of chat history
    with st.sidebar:
        st.markdown("### 💾 Chatverlauf exportieren")
        
        if st.button("Chat exportieren"):
            os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)  # create folder if not exists
            
            # Filename with timestamp and path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_{timestamp}.json"
            path = os.path.join(CHAT_HISTORY_DIR, filename)
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=4)
            st.success(f"Chatverlauf als {filename} exportiert!")
        
    # -------------------------------
    # Load only new or changed PDFs
    # -------------------------------
    if "pdfs_loaded" not in st.session_state:
        st.session_state.pdfs_loaded = False
        
    if not st.session_state.pdfs_loaded:
        all_documents = load_documents()
        st.write(f"Loaded {len(all_documents)} PDFs. Processing…")
        
        progress_bar = st.progress(0)
        
        stored_hashes = load_hashes()
        documents_to_process = []

        for i, doc in enumerate(all_documents):
            source = doc.metadata.get("source")
            filepath = os.path.join(PDF_FOLDER, os.path.basename(source))
            file_hash = compute_md5(filepath)
            if stored_hashes.get(os.path.basename(source)) != file_hash:
                documents_to_process.append(doc)
            progress_bar.progress((i + 1) / len(all_documents))


        if documents_to_process:
            st.write(f"{len(documents_to_process)} PDFs are new/changed. Splitting into chunks…")
            chunks = split_documents(documents_to_process)  # split only new/changed docs
            
            st.write("Adding embeddings to Chroma…")
            add_to_chroma(chunks)
            st.success(f"Processed {len(documents_to_process)} new/changed PDFs and added {len(chunks)} chunks.")
        else:
            
            st.success("✅ No new or changed PDFs detected.")
        st.session_state.pdfs_loaded = True
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = _build_rag_chain(llm_model_name=llm_model_name)
        st.success("RAG chain is ready.")

    rag_chain = st.session_state.rag_chain
    print("RAG chain is ready.")
    # Initialize chat history in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display the full chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**You:** {chat['question']}")
        with st.chat_message("ai"):
            st.markdown(f"**RAG:** {chat['answer']}")

    # Input field for new user question
    question = st.chat_input("Ask a question about the PDFs:")
    if not question:
        return

    # Show the user's question in the chat
    with st.chat_message("user"):
        st.markdown(f"**You:** {question}")
    # Show the AI's answer in the chat
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            try:
                # Format chat history for the prompt
                chat_history_str = format_chat_history(st.session_state.chat_history)
                # Invoke the RAG chain with the question and chat history
                response = rag_chain.invoke(
                    {"question": question, "chat_history": chat_history_str}
                )
                st.markdown("### Answer")
                st.success(response)
                # Append the new Q&A to the chat history and save it
                st.session_state.chat_history.append(
                    {"question": question, "answer": response}
                )
                # save_chat_history(st.session_state.chat_history, CHAT_HISTORY_DIR)
            except Exception as e:
                st.error(f"Error processing your question: {e}")

if __name__ == "__main__":
    main()
