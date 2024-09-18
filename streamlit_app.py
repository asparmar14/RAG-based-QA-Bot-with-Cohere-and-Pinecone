import streamlit as st
import PyPDF2
from io import BytesIO
from backend import qa_pipeline, retrieve_relevant_chunks  # Import your backend functions here

def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

st.title("Interactive QA Bot")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    document_text = process_pdf(uploaded_file)
    st.write("Document uploaded and text extracted.")
    
    question = st.text_input("Ask a question based on the uploaded document")
    
    if question:
        answer = qa_pipeline(document_text, question)
        st.write("### Answer:")
        st.write(answer)
        st.write("### Retrieved Chunks:")
        for chunk in retrieve_relevant_chunks(question):
            st.write(f"Score: {chunk['score']}, Text: {chunk['metadata']['text']}")