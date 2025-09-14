@echo off
cd /d "C:\Users\test\Masterthesis_PDF_Quotes_RAG"
call .venv\Scripts\activate
streamlit run RAG.py
pause