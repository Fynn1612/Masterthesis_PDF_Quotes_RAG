from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_pdf_text(uploaded_files):
  """
  Extracts text from all pages of uploaded PDF files.

  Example:
  - Page 1: "Hello from page one."
  - Page 2: "This is page two."
  - Result: "Hello from page one.This is page two."
  """
  text = ""
  for file in uploaded_files:
    reader = PdfReader(file)
    for page in reader.pages:
      text += page.extract_text() or ""
  return text

def get_text_chunks(text):
  """
  Splits large text into overlapping chunks for LLM context.

  Example:
  If text = "abcdefg", chunk_size=4, overlap=2:
  - ["abcd", "cdef", "efg"]
  """
  splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
  return splitter.split_text(text)
