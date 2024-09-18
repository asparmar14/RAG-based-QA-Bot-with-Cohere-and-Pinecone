import os
import cohere
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Pinecone API setup
PINECONE_API_KEY = 'your-api-key'
PINECONE_ENV = 'us-east-1'  

# Create Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Model for document embeddings (384 dimensions)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone index with 384 dimensions
index_name = 'qa-bot-index'

# Check if the index exists, otherwise create it
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name) # Delete the index if it exists

pc.create_index(
    name=index_name,
    dimension=384,  # Match the model's output dimension
    metric="cosine",  # Use cosine similarity for matching
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Access the index
index = pc.Index(index_name)

def embed_text(text):
    """Embed text using the sentence transformer model."""
    return model.encode([text], convert_to_tensor=False).tolist()[0]

def add_document_to_pinecone(document_id, document_text):
    """Add document chunks to Pinecone index."""
    chunk_size = 300
    chunks = [document_text[i:i+chunk_size] for i in range(0, len(document_text), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        index.upsert([(f'{document_id}-{i}', embedding, {'text': chunk})])

def retrieve_relevant_chunks(question, top_k=3):
    """Retrieve the most similar chunks from Pinecone based on the question."""
    query_embedding = embed_text(question)
    result = index.query(vector=[query_embedding], top_k=top_k, include_metadata=True)
    return result['matches']

# Initialize Cohere API client
COHERE_API_KEY = 'your-api-key'
co = cohere.Client(COHERE_API_KEY)

def generate_answer(question, relevant_chunks):
    """Generate an answer to the question based on relevant document chunks."""
    context = " ".join([chunk['metadata']['text'] for chunk in relevant_chunks])
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    
    return response.generations[0].text.strip()

def qa_pipeline(document_text, question):
    """Run the QA pipeline: add document, retrieve chunks, generate answer."""
    document_id = "doc1"
    add_document_to_pinecone(document_id, document_text)
    relevant_chunks = retrieve_relevant_chunks(question)
    answer = generate_answer(question, relevant_chunks)
    return answer

