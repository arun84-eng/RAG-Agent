import streamlit as st
import math
import re

# Set page configuration
st.set_page_config(page_title="RAG-Powered Q&A Assistant", page_icon="🤖")
st.title("RAG-Powered Multi-Agent Q&A Assistant")

# --- DOCUMENT MANAGEMENT ---

# Initialize session state for documents
if 'documents' not in st.session_state:
    st.session_state.documents = []
    st.session_state.initialized = False

# Document Management Section
st.header("Document Management")

# File uploader for custom documents
uploaded_files = st.file_uploader("Upload your own documents (optional)", 
                                accept_multiple_files=True, 
                                type=['txt', 'md', 'pdf'])

# Sample docs selection
use_sample_docs = st.checkbox("Use sample documents", value=True)

# Process document uploads
if uploaded_files:
    new_docs = []
    for file in uploaded_files:
        try:
            content = file.getvalue().decode('utf-8')
            new_docs.append({
                "name": file.name,
                "content": content
            })
            st.success(f"Successfully processed: {file.name}")
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    st.session_state.documents = new_docs
    st.session_state.initialized = False

# Create sample documents if requested
if use_sample_docs:
    if not st.session_state.documents:
        sample_docs = [
            {
                "name": "company_faq.txt",
                "content": """# Company FAQ
                
## General Questions

### What is RAG-Powered Assistant?
RAG-Powered Assistant is a cutting-edge AI system that combines retrieval-augmented generation with a multi-agent approach to provide accurate answers based on your documents.

### How does the system work?
Our system uses a two-step process: first, it retrieves relevant information from your documents, then it generates natural language answers using an advanced language model. The system can also route specialized queries to dedicated tools."""
            },
            {
                "name": "product_specs.txt",
                "content": """# RAG-Assistant Technical Specifications

## System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space
- Internet connection for LLM API access

## Core Components

### Document Processor
- Handles document ingestion and chunking
- Supports TXT, MD, and PDF formats
- Optimal chunk size: 512 tokens with 128 token overlap"""
            }
        ]
        st.session_state.documents = sample_docs
        st.info(f"Loaded {len(sample_docs)} sample documents")

# Initialize system button
if st.session_state.documents and not st.session_state.initialized:
    if st.button("Initialize System"):
        st.session_state.initialized = True
        st.success("System initialized successfully!")
elif st.session_state.initialized and st.button("Reinitialize System"):
    st.session_state.initialized = True
    st.success("System reinitialized successfully!")

# --- CALCULATOR FUNCTION ---

def calculate(query):
    """Extract and calculate mathematical expression from query"""
    # Clean the query
    clean_query = query.lower()
    clean_query = re.sub(r'calculate\s+', '', clean_query)
    clean_query = re.sub(r'what\s+is\s+', '', clean_query)
    clean_query = re.sub(r'the\s+', '', clean_query)
    
    # Handle square root
    if 'square root' in clean_query:
        match = re.search(r'square\s+root\s+of\s+(\d+\.?\d*)', clean_query)
        if match:
            num = float(match.group(1))
            result = math.sqrt(num)
            return f"The square root of {num} is {result:.4f}"
    
    # Handle basic arithmetic
    try:
        # Extract numbers and operators
        expression = re.sub(r'[^0-9+\-*/().\s]', '', clean_query)
        expression = expression.strip()
        
        # Safe evaluation
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except:
        return "I couldn't parse this calculation. Please try a simpler expression."

# --- DOCUMENT SEARCH ---

def search_documents(query):
    """Search for relevant content in documents"""
    if not st.session_state.documents:
        return "No documents available to search. Please upload documents or use sample documents."
    
    query_terms = query.lower().split()
    # Remove common words
    stop_words = {'the', 'a', 'an', 'and', 'is', 'in', 'of', 'to', 'for'}
    query_terms = [term for term in query_terms if term not in stop_words]
    
    results = []
    
    for doc in st.session_state.documents:
        content = doc["content"].lower()
        
        # Check if any query term appears in the document
        if any(term in content for term in query_terms):
            # Find the most relevant paragraph
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if any(term in para for term in query_terms):
                    results.append({
                        "source": doc["name"],
                        "text": para.strip()
                    })
                    break  # Take only the first matching paragraph per document
    
    if results:
        answer = "Here's what I found in the documents:\n\n"
        for i, result in enumerate(results):
            answer += f"From {result['source']}:\n{result['text']}\n\n"
        return answer
    else:
        return "I couldn't find relevant information in the documents."

# --- TOOL SELECTION ---

def route_query(query):
    """Determine which tool to use"""
    query_lower = query.lower()
    
    # Check for calculation patterns
    if ('calculate' in query_lower or 
        'what is' in query_lower and any(c.isdigit() for c in query_lower) or
        'square root' in query_lower or
        any(op in query_lower for op in ['+', '-', '*', '/', '^'])):
        return "calculator"
    
    # Otherwise use document search
    return "document_search"

# --- MAIN Q&A INTERFACE ---

st.header("Ask a Question")

# Show warning if system not initialized
if not st.session_state.initialized:
    st.warning("Please initialize the system first.")

# Query input
user_query = st.text_input("Enter your question:", placeholder="e.g., What are the product features? or calculate the square root of 88")

# Process the query
if user_query and st.session_state.initialized:
    with st.spinner("Processing your question..."):
        # Determine which tool to use
        tool = route_query(user_query)
        
        # Display tool information
        st.subheader("Agent Workflow")
        st.markdown("**Tool Selected:**")
        st.info(tool)
        
        # Process with appropriate tool
        if tool == "calculator":
            answer = calculate(user_query)
        else:
            answer = search_documents(user_query)
        
        # Display answer
        st.subheader("Answer")
        st.write(answer)

# Footer with instructions
st.markdown("---")
st.markdown("""
**Instructions:**
- Upload your own documents or use the sample documents
- Click 'Initialize System' after document selection
- Ask questions about the content of the documents
- Try calculation questions, e.g., "what is the square root of 88?"
""")