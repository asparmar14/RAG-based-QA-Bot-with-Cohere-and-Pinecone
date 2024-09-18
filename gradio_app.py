import gradio as gr
import PyPDF2
from backend import qa_pipeline, retrieve_relevant_chunks

def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_answer(file, question):
    document_text = process_pdf(file)
    answer = qa_pipeline(document_text, question)
    relevant_chunks = retrieve_relevant_chunks(question)
    
    # Convert relevant_chunks to a list of dicts for DataFrame
    chunks_data = [
        {"Score": chunk["score"], "Text": chunk["metadata"]["text"]}
        for chunk in relevant_chunks
    ]
    
    return answer, chunks_data

interface = gr.Interface(
    fn=get_answer,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a question")],
    outputs=[gr.Textbox(label="Answer"), gr.Dataframe(headers=["Score", "Text"], label="Retrieved Chunks")],
    title="Interactive QA Bot"
)

interface.launch()
