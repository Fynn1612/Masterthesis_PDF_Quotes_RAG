"""
Developer Mode Module
---------------------

Provides developer/debugging utilities to inspect Chroma Vector Store.
Displays document count and allows testing similarity queries from sidebar.
"""

import streamlit as st


def inspect_vectorstore(vectorstore):
  """
  Displays basic info and tools for inspecting Chroma vectorstore in sidebar.

  Parameters:
  - vectorstore (Chroma): The active LangChain Chroma vectorstore instance
  """
  with st.sidebar.expander("🧪 ChromaDB Inspector", expanded=False):
    # Show document count
    try:
      doc_count = vectorstore._collection.count()
      st.success(f"🔎 {doc_count} documents stored in ChromaDB.")
      docs = vectorstore.get(include=["metadatas", "documents"])
      for i, (docs, meta) in enumerate(zip(docs["documents"], docs["metadatas"])):
        st.markdown(f"**Eintrag {i + 1}:**")
        st.markdown(f"- Quelle: {meta.get('source')}, Seite: {meta.get('page')}")
        st.markdown(f"- Textauszug: {docs[:100]} ...")
        st.markdown("---")
    except Exception as e:
      st.error("⚠️ Could not fetch document count.")
      st.code(str(e))

    # Input box for similarity query
    query = st.text_input("🔍 Test a query against ChromaDB")

    if query:
      try:
        results = vectorstore.similarity_search(query, k=3)
        if results:
          st.markdown("### 🔎 Top Matching Chunks:")
          for i, doc in enumerate(results):
            content = getattr(doc, "page_content", str(doc))[:300]
            st.markdown(f"**Result {i + 1}:**\n\n{content}...")
            st.markdown("---")
        else:
          st.info("No matching chunks found.")
      except Exception as e:
        st.error("❌ Error querying ChromaDB")
        st.code(str(e))
