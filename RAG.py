from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

from utils.config import GOOGLE_API_KEY

#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st

from utils.meta_data_handling import get_pdf_metadata, load_stored_metadata, save_metadata

PDF_FOLDER = "G:/Meine Ablage/Masterarbeit_RAG_PDFs/PDFs"
METADATA_PATH = "G:/Meine Ablage/Masterarbeit_RAG_PDFs/pdf_metadata.json"
VECTORSTORE_PATH = "chroma_db"

PROJECT_NAME = "Masterarbeit_RAG_PDFs"

GOOGLE_DRIVE_BASE = r"G:\Meine Ablage"

folder_path = PDF_FOLDER

llm_model_name = "gemini-2.5-flash"

def get_embedding():
    """Get the embedding model instance.

    Returns:
        GoogleGenerativeAIEmbeddings: The embedding model instance.
    """
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)


@st.cache_resource
def load_pdf():
    """Load PDF documents from the specified folder and extract metadata.
    
    Returns:
        tuple: A tuple containing the current metadata, if there were changes and a list of loaded documents.
    """

    current_metadata = get_pdf_metadata(folder_path)
    stored_metadata = load_stored_metadata(METADATA_PATH)
    
    if current_metadata != stored_metadata:
        print("[✓] New metadata found.")
        documents = []
        # find new PDFs
        new_pdfs = []
        #print(f"Current metadata: {current_metadata}")
        #print(f"Stored metadata: {stored_metadata}")

        for  file_name,  last_modified in current_metadata.items():
            # last_modified = meta["last_modified"]
            #print(file_name)
            #print(last_modified)
            if file_name not in stored_metadata:
                new_pdfs.append(file_name)
                print(f"[DEBUG] {file_name} ist NEU (nicht in stored_metadata).")
            elif last_modified != stored_metadata[file_name]:
                new_pdfs.append(file_name)
                print(f"[DEBUG] {file_name} wurde GEÄNDERT (last_modified unterschiedlich). {last_modified} im vergleich zu {stored_metadata[file_name]["last_modified"]}")
                #print(f"[✓] New PDF found: {file_name}")
        if new_pdfs:
            for pdf in new_pdfs:
                print(f"Loading new PDF: {pdf}")
                pdf_loader = UnstructuredPDFLoader(file_path=os.path.join(folder_path, pdf))
                document = pdf_loader.load()
                documents.append(document)
        changes = True

        return changes, current_metadata, documents

    else:
        changes = False

        return changes, current_metadata, None

@st.cache_resource
def _build_rag_chain(documents, llm_model_name = llm_model_name, changes=False, current_metadata=None):
    """
    

    Args:
        documents (list): List of newly uploaded or changed documents to be processed.
        llm_model_name (str): Name of the LLM model to use for the RAG chain.
        changes (bool): Flag indicating if there are new or updated documents.
        current_metadata (dict): Metadata of the current set of documents.

    Returns:
        rag_chain: The RAG chain for document retrieval and question answering.
    """
    
    # Get embedding model
    embedding = get_embedding()
    
    if changes:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        doc_chunks = []
        for d in documents: 
            chunks = splitter.split_documents(d)
            for chunk in chunks:
                print(chunk.metadata["source"])
            doc_chunks.extend(chunks)
        #print(f" doc_chunks: {doc_chunks}")
        print(f"[✓] Document split into {len(doc_chunks)} chunks.")

        print("Pulling embedding model...")
                                
        persist_path = os.path.join(GOOGLE_DRIVE_BASE, PROJECT_NAME, "Vectorstores", "gemini", "chroma_db")
        os.makedirs(persist_path, exist_ok=True)
                
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=persist_path,
            )
        
        vector_store.add_documents(documents=doc_chunks)

        save_metadata(METADATA_PATH, current_metadata)
        print("[✓] Chunks embedded and stored in vector database.")

    elif not changes:
    # If metadata is unchanged, load existing vectorstore
        persist_path = os.path.join(GOOGLE_DRIVE_BASE, PROJECT_NAME, "Vectorstores", "gemini", "chroma_db")
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=persist_path,
        )

    llm = ChatGoogleGenerativeAI(model=llm_model_name, api_key=GOOGLE_API_KEY)

    multi_query_prompt = PromptTemplate(
        input_variables = ["question"],
        template = ("You are an AI assistant. Generate 5 different versions of the user question "
            "to improve document retrieval from a vector database.\n"
            "Original question: {question}"
            
        )
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(),
        llm=llm,
        prompt=multi_query_prompt,
    )

    print("[✓] Retriever with multi-query setup complete.")

    context_prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the context below:\n{context}\n\nQuestion: {question}. Provide the source(filename) and the page number of the chunks you used."
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | context_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def main():
    st.set_page_config(page_title="Masterthesis Literature Chatbot", layout="centered")
    st.title("Masterthesis Literature Chatbot")
    st.caption("Chat with multiple PDFs :books:")
    
    # Initialize required Streamlit session state variables
    setup_session_state()
    
    # Allow user to download the chat history as CSV
    if st.session_state.chat_history:
        render_download_chat_history()
        
    # Render previous chat messages (Q&A)
    if st.session_state.get("chat_history", []):
        render_chat_history()

    # Load new or updated PDFs
    changes, current_metadata, documents = load_pdf()
    if not documents:
        st.info("No new or updated PDFs found. Using existing vectorstore.")

    # Build RAG chain
    rag_chain = _build_rag_chain(documents, llm_model_name=llm_model_name, changes=changes, current_metadata=current_metadata)

    # Create a text input for user questions
    question = st.chat_input("Ask a question about the PDFs:")
    if not question:
        return
    
    with st.chat_message("user"):
        st.markdown(f"**You:** {question}")
    with st.chat_message("ai"):
        
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke(question)
                st.markdown("### Answer")
                st.success(response)
                st.session_state.chat_history.append({"question": question, "answer": response})
        
            except Exception as e:
                st.error(f"Error processing your question: {e}")


if __name__ == "__main__":
    main()