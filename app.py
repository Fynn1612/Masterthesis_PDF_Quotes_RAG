import streamlit as st

from utils.chat_handler import (
  setup_session_state,
  render_chat_history,
  render_download_chat_history,
  handle_user_input,
  render_uploaded_files_expander
)
from utils.sidebar_handler import (
  render_model_selector,
  sidebar_file_upload,
  sidebar_provider_change_check,
  sidebar_utilities
)
from utils.developer_mode import inspect_vectorstore
from utils.llm_handler import get_llm_chain


def main():
  """
  Main entry point for the Masterthesis Literature Chatbot Streamlit app.

  This function:
  - Sets up the page configuration
  - Handles model and file selection in the sidebar
  - Manages PDF upload and vector store creation
  - Renders chat interface and handles user questions
  - Allows downloading chat history
  """
  st.set_page_config(page_title="Masterthesis Literature Chatbot", layout="centered")
  st.title("Masterthesis Literature Chatbot")
  st.caption("Chat with multiple PDFs :books:")

  # Initialize required Streamlit session state variables
  setup_session_state()

  # Allow user to download the chat history as CSV
  if st.session_state.chat_history:
    render_download_chat_history()

  # Sidebar configuration: model selection, file upload, provider recheck
  with st.sidebar:
    with st.expander("⚙️ Configuration", expanded=True):
      model_provider, model = render_model_selector()
      sidebar_file_upload(model_provider)
      sidebar_provider_change_check(model_provider, model)

    # Utility buttons: reset, clear, undo
    sidebar_utilities()

  # Show message if no files are uploaded yet
  if not st.session_state.get(f"uploaded_files_{st.session_state.uploader_key}", []):
    st.info("📄 Please upload and submit PDFs to start chatting.")

  # Warn if new PDFs are uploaded but not submitted
  if st.session_state.get("unsubmitted_files", False):
    st.warning("📄 New PDFs uploaded. Please submit before chatting.")

  # Show uploaded files summary
  if st.session_state.get("vector_store", None) and st.session_state.get("pdf_files", []):
    render_uploaded_files_expander()

  # Render previous chat messages (Q&A)
  if st.session_state.get("chat_history", []):
    render_chat_history()

  # Show chat input box and process question with selected LLM
  if st.session_state.get("vector_store"):
    handle_user_input(model_provider, model, get_llm_chain(model_provider, model, st.session_state.get("vector_store")))

  # Developer mode: inspect Chroma vectorstore
  if st.session_state.vector_store:
    inspect_vectorstore(st.session_state.vector_store)

if __name__ == "__main__":
  main()
