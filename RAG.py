import streamlit as st
import os

# Import LangChain and utility modules
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.chat_history_handling import (
    format_chat_history,
    save_chat_history,
    list_chat_histories,
    load_chat_history,
)
from utils.meta_data_handling import (
    get_pdf_metadata,
    load_stored_metadata,
    save_metadata,
)
from utils.config import GOOGLE_API_KEY

# Define constants for file paths and configuration
PDF_FOLDER = "G:/Meine Ablage/Masterarbeit_RAG_PDFs/PDFs"
METADATA_PATH = "G:/Meine Ablage/Masterarbeit_RAG_PDFs/pdf_metadata.json"
CHAT_HISTORY_DIR = "chat_histories"
PROJECT_NAME = "Masterarbeit_RAG_PDFs"
GOOGLE_DRIVE_BASE = r"G:\Meine Ablage"
llm_model_name = "gemini-2.5-flash"


def get_embedding():
    """
    Ensure an event loop exists and return the Google Generative AI Embedding model.
    This is needed because the embedding model uses async code and Streamlit runs in its own thread.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_API_KEY
    )


@st.cache_resource
def load_pdf():
    """
    Loads PDF documents from the specified folder and checks for new or updated files.
    Only new or changed PDFs are loaded and processed.
    Returns:
        changes (bool): Whether there are new/changed PDFs.
        current_metadata (dict): Metadata of all PDFs in the folder.
        documents (list): List of loaded documents (if any new/changed).
    """
    current_metadata = get_pdf_metadata(PDF_FOLDER)
    stored_metadata = load_stored_metadata(METADATA_PATH)

    if current_metadata != stored_metadata:
        print("[✓] New metadata found.")
        documents = []
        new_pdfs = []
        # Identify new or changed PDFs by comparing metadata
        for file_name, last_modified in current_metadata.items():
            if file_name not in stored_metadata:
                new_pdfs.append(file_name)
            elif last_modified != stored_metadata[file_name]:
                new_pdfs.append(file_name)
                print(f"[✓] New PDF found: {file_name}")
        # Load only new/changed PDFs
        if new_pdfs:
            for pdf in new_pdfs:
                print(f"Loading new PDF: {pdf}")
                pdf_loader = UnstructuredPDFLoader(
                    file_path=os.path.join(PDF_FOLDER, pdf),
                    mode = "elements",
                    unstructured_kwargs={"strategy": "fast"},
                )
                document = pdf_loader.load()
                documents.append(document)
        changes = True
        return changes, current_metadata, documents
    else:
        # No changes, so no need to reload documents
        changes = False
        return changes, current_metadata, None


@st.cache_resource
def _build_rag_chain(
    documents, llm_model_name=llm_model_name, changes=False, current_metadata=None
):
    """
    Builds the RAG (Retrieval-Augmented Generation) chain for answering questions based on PDF content.
    If there are new/changed documents, splits and embeds them, otherwise loads the existing vectorstore.
    Returns the complete RAG chain.
    """
    # Get embedding model (ensures event loop)
    embedding = get_embedding()

    if changes:
        # If there are new/changed documents, split them into chunks and embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        doc_chunks = []
        for d in documents:
            chunks = splitter.split_documents(d)
            for chunk in chunks:
                print(chunk.metadata["source"])
            doc_chunks.extend(chunks)
        print(f"[✓] Document split into {len(doc_chunks)} chunks.")

        print("Pulling embedding model...")
        # Define persistent path for the vectorstore
        persist_path = os.path.join(
            GOOGLE_DRIVE_BASE, PROJECT_NAME, "Vectorstores", "gemini", "chroma_db"
        )
        os.makedirs(persist_path, exist_ok=True)
        # Create new vectorstore and add new chunks
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=persist_path,
        )
               
        vector_store.add_documents(documents=doc_chunks)
        # Save updated metadata so we know which PDFs are already processed
        save_metadata(METADATA_PATH, current_metadata)
        print("[✓] Chunks embedded and stored in vector database.")
    elif not changes:
        # If no changes, just load the existing vectorstore
        persist_path = os.path.join(
            GOOGLE_DRIVE_BASE, PROJECT_NAME, "Vectorstores", "gemini", "chroma_db"
        )
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=persist_path,
        )

    # Load the LLM (Google Gemini)
    llm = ChatGoogleGenerativeAI(model=llm_model_name, api_key=GOOGLE_API_KEY)

    # Multi-query prompt: helps retrieve more relevant chunks by rephrasing the question
    multi_query_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are an AI assistant. Generate 10 different versions of the user question "
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
                "context": retriever,
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

    # Sidebar: List and load old chat histories
    with st.sidebar:
        st.markdown("### 💬 Alte Chats")
        chat_files = list_chat_histories(CHAT_HISTORY_DIR)
        selected_chat = st.selectbox(
            "Wähle einen Chatverlauf", ["Neuer Chat"] + chat_files
        )
        if selected_chat != "Neuer Chat":
            # Load selected chat history into session state
            st.session_state.chat_history = load_chat_history(
                selected_chat, CHAT_HISTORY_DIR
            )
            st.success(f"Chat {selected_chat} geladen!")

    # Sidebar: Button to manually save current chat history
    with st.sidebar:
        if st.button("Chatverlauf speichern"):
            save_chat_history(st.session_state.chat_history, CHAT_HISTORY_DIR)
            st.success("Chatverlauf gespeichert!")

    # Load PDFs and check for changes
    changes, current_metadata, documents = load_pdf()
    if not documents:
        st.info("No new or updated PDFs found. Using existing vectorstore.")

    # Build the RAG chain (retriever + LLM)
    rag_chain = _build_rag_chain(
        documents,
        llm_model_name=llm_model_name,
        changes=changes,
        current_metadata=current_metadata,
    )

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
                save_chat_history(st.session_state.chat_history, CHAT_HISTORY_DIR)
            except Exception as e:
                st.error(f"Error processing your question: {e}")


if __name__ == "__main__":
    main()
