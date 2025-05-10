import os
import re
from typing import List, Dict, Any

def create_sample_documents():
    """Create sample documents with example content"""
    if not os.path.exists("sample_docs"):
        os.makedirs("sample_docs")
    
    # Sample company FAQ
    company_faq = """
    # Company FAQ
    
    ## General Questions
    
    ### What is RAG-Powered Assistant?
    RAG-Powered Assistant is a cutting-edge AI system that combines retrieval-augmented generation with a multi-agent approach to provide accurate answers based on your documents.
    
    ### How does the system work?
    Our system uses a two-step process: first, it retrieves relevant information from your documents, then it generates natural language answers using an advanced language model. The system can also route specialized queries to dedicated tools.
    
    ## Technical Support
    
    ### What file formats are supported?
    Currently, we support TXT, MD, and PDF file formats for document ingestion.
    
    ### How secure is my data?
    Your data is processed locally and is not stored on external servers. The vector embeddings are kept in memory for the duration of your session only.
    
    ## Pricing and Plans
    
    ### Is there a free tier?
    Yes, the basic version is available for free with limited document processing. Premium plans start at $19.99/month.
    
    ### What are the limits of document processing?
    The free tier allows processing up to 5 documents with a maximum of 50 pages each. Premium plans remove these restrictions.
    """
    
    # Sample product specs
    product_specs = """
    # RAG-Assistant Technical Specifications
    
    ## System Requirements
    
    - Python 3.8 or higher
    - 8GB RAM minimum (16GB recommended)
    - 2GB free disk space
    - Internet connection for LLM API access
    
    ## Core Components
    
    ### Document Processor
    - Handles document ingestion and chunking
    - Supports TXT, MD, and PDF formats
    - Optimal chunk size: 512 tokens with 128 token overlap
    
    ### Vector Store
    - Uses FAISS for efficient similarity search
    - Dimensions: 1536 (OpenAI ada-002 embeddings)
    - Top-k retrieval with MMR reranking
    
    ### Agent System
    - Tool routing based on query classification
    - Available tools: Calculator, Dictionary, RAG pipeline
    - Custom tool integration available in Premium plans
    
    ## Performance Metrics
    
    - Average query response time: <2 seconds
    - Retrieval precision: 85% (based on benchmark tests)
    - Agent routing accuracy: 92%
    """
    
    # Sample user manual
    user_manual = """
    # RAG-Assistant User Manual
    
    ## Getting Started
    
    ### Document Upload
    1. Navigate to the Document Management section
    2. Click "Upload your own documents"
    3. Select one or more files to upload
    4. Wait for processing to complete
    
    ### Using Sample Documents
    If you want to test the system without uploading your own documents:
    1. Check the "Use sample documents" option
    2. Click "Initialize System"
    
    ## Asking Questions
    
    Enter your question in the text input field and press Enter. The system will:
    1. Analyze your query
    2. Route it to the appropriate tool or retrieval system
    3. Generate and display the answer
    
    ### Query Types
    
    #### Document Questions
    Ask anything about the content in your documents:
    - "What are the key features of the product?"
    - "How do I upload documents?"
    
    #### Calculations
    Use the calculator by including the keyword "calculate":
    - "Calculate the square root of 144"
    - "Calculate 25% of 80"
    
    #### Definitions
    Get definitions by using the keyword "define":
    - "Define machine learning"
    - "Define vector database"
    
    ## Troubleshooting
    
    ### System Not Responding
    If the system appears frozen, try:
    1. Refreshing the page
    2. Reinitializing the system
    
    ### Incorrect Answers
    If answers seem incorrect:
    1. Check if your documents contain the relevant information
    2. Try rephrasing your question
    3. Ensure the system was properly initialized
    """
    
    # Write the sample documents to files
    with open("sample_docs/company_faq.txt", "w") as f:
        f.write(company_faq)
    
    with open("sample_docs/product_specs.txt", "w") as f:
        f.write(product_specs)
    
    with open("sample_docs/user_manual.txt", "w") as f:
        f.write(user_manual)

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks of approximately the specified size.
    
    Args:
        text: The text to split into chunks
        chunk_size: The target size for each chunk
        chunk_overlap: The overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If text is shorter than chunk size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of the chunk
        end = start + chunk_size
        
        # If we're at the end of the text, just use the rest
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find a good breaking point
        # Look for the last period, question mark, or exclamation point followed by a space
        last_sentence_break = max(
            text.rfind('. ', start, end),
            text.rfind('? ', start, end),
            text.rfind('! ', start, end)
        )
        
        # If no good breaking point, look for the last space
        if last_sentence_break == -1:
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                end = last_space
            # If no space found, just break at chunk_size
        else:
            # Include the punctuation and space
            end = last_sentence_break + 2
        
        # Add the chunk
        chunks.append(text[start:end])
        
        # Move to the next chunk, accounting for overlap
        start = end - chunk_overlap
    
    return chunks

def process_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a document file, chunking it and preparing for vector store.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of document chunks with metadata
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Get basic document metadata
        doc_name = os.path.basename(file_path)
        
        # Chunk the document
        text_chunks = chunk_text(content)
        
        # Create document chunks with metadata
        document_chunks = []
        for i, chunk in enumerate(text_chunks):
            document_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": doc_name,
                    "chunk_id": i,
                    "total_chunks": len(text_chunks)
                }
            })
        
        return document_chunks
    
    except Exception as e:
        raise Exception(f"Error processing document {file_path}: {str(e)}")

if __name__ == "__main__":
    create_sample_documents()
