from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

from utils.config import GOOGLE_API_KEY, GROQ_API_KEY

from langchain.prompts import ChatPromptTemplate
#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import traceback

from meta_data_handling import get_pdf_metadata, load_stored_metadata, save_metadata

try:

    PDF_FOLDER = "G:/Meine Ablage/Masterarbeit_RAG_PDFs/PDFs"
    METADATA_PATH = "G:/Meine Ablage/Masterarbeit_RAG_PDFs/pdf_metadata.json"
    VECTORSTORE_PATH = "chroma_db"
    
    PROJECT_NAME = "Masterarbeit_RAG_PDFs"

    GOOGLE_DRIVE_BASE = r"G:\Meine Ablage"

    folder_path = PDF_FOLDER
    
    current_metadata = get_pdf_metadata(folder_path)
    stored_metadata = load_stored_metadata(METADATA_PATH)
    
    llm_model_name = "gemini-2.5-flash"
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    if current_metadata != stored_metadata:
        print("[✓] New metadata found.")
        documents = []
        # find new PDFs
        new_pdfs = []
        print(f"Current metadata: {current_metadata}")
        print(f"Stored metadata: {stored_metadata}")

        for  file_name,  last_modified in current_metadata.items():
           # last_modified = meta["last_modified"]
            print(file_name)
            print(last_modified)
            if file_name not in stored_metadata:
                new_pdfs.append(file_name)
                print(f"[DEBUG] {file_name} ist NEU (nicht in stored_metadata).")
            elif last_modified != stored_metadata[file_name]["last_modified"]:
                new_pdfs.append(file_name)
                print(f"[DEBUG] {file_name} wurde GEÄNDERT (last_modified unterschiedlich). {last_modified} im vergleich zu {stored_metadata[file_name]["last_modified"]}")
                #print(f"[✓] New PDF found: {file_name}")
        if new_pdfs:
            for pdf in new_pdfs:
                print(f"Loading new PDF: {pdf}")
                pdf_loader = UnstructuredPDFLoader(file_path=os.path.join(folder_path, pdf))
                document = pdf_loader.load()
                documents.append(document)
        
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            doc_chunks = []
            for d in documents: 
                doc = splitter.split_documents(d)
                doc_chunks.extend(doc)
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
            
            #vector_store.persist()
            
        # else:
        # # Otherwise, create a new vectorstore from the chunks
        #     vector_store = Chroma.from_texts(
        #         texts=[c["text"] for c in chunks],
        #         embedding=embedding,
        #         metadatas=[c["metadata"] for c in chunks],
        #         persist_directory=persist_path
        #     )
        #     vectorstore.persist()
        save_metadata(METADATA_PATH, current_metadata)
        print("[✓] Chunks embedded and stored in vector database.")
    else: 
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
        "Answer the question using ONLY the context below:\n{context}\n\nQuestion: {question}"
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | context_prompt
        | llm
        | StrOutputParser()
    )

    # === Example Query ===
    user_question = "What is epistemic uncertainty?"
    response = rag_chain.invoke(user_question)

    print("\n=== RAG Response ===\n")
    print(response)


except Exception:
    traceback.print_exc()