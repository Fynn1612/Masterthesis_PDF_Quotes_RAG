from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_pdf_text(uploaded_files):
  """
  Extracts text from all pages of uploaded PDF files and returns a list of page-wise text chunks
    with metadata (filename + page number).

    Returns:
      list of dicts:
        [
          {"text": "...", "metadata": {"source": "file.pdf", "page": 1}},
          {"text": "...", "metadata": {"source": "file.pdf", "page": 2}},
          ...
        ]
    """
  text_with_meta = []
  for file in uploaded_files:
    reader = PdfReader(file)
    filename = getattr(file, "name", str(file))
    for i, page in enumerate(reader.pages, start = 1):
      page_text = page.extract_text() or ""
      if page_text.strip():  # Only add non-empty pages
        text_with_meta.append({"text": page_text, "metadata": {"source": filename, "page": i}})

  return text_with_meta

def get_text_chunks(docs_with_metadata, chunk_size=5000, chunk_overlap=500):
    """
    Splits texts from PDFs into overlapping chunks while keeping metadata.

    Args:
        docs_with_metadata: list of dicts, each dict has:
            {
              "text": "full page text",
              "metadata": {"source": "file.pdf", "page": 1}
            }
        chunk_size: max characters per chunk
        chunk_overlap: overlap between chunks

    Returns:
        list of dicts:
            [
              {"text": "chunk1", "metadata": {"source": "file.pdf", "page": 1}},
              {"text": "chunk2", "metadata": {"source": "file.pdf", "page": 1}},
              ...
            ]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks_with_metadata = []

    for doc in docs_with_metadata:
        text_chunks = splitter.split_text(doc["text"])
        for chunk in text_chunks:
            chunks_with_metadata.append({
                "text": chunk,
                "metadata": doc["metadata"]
            })

    return chunks_with_metadata
