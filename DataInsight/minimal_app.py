import streamlit as st
import math
import re
import os

# Set page configuration
st.set_page_config(page_title="Multi-Tool Q&A System", page_icon="🤖")
st.title("Multi-Tool Q&A System")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Document Upload Section
st.sidebar.header("Document Upload")
uploaded_files = st.sidebar.file_uploader("Upload your documents", 
                                accept_multiple_files=True,
                                type=['txt', 'md'])

# Process uploaded files
if uploaded_files:
    st.session_state.documents = []
    for file in uploaded_files:
        try:
            content = file.getvalue().decode('utf-8')
            st.session_state.documents.append({
                "name": file.name,
                "content": content
            })
            st.sidebar.success(f"Uploaded: {file.name}")
        except Exception as e:
            st.sidebar.error(f"Error processing {file.name}: {str(e)}")

# Display number of documents loaded
if st.session_state.documents:
    st.sidebar.info(f"Loaded {len(st.session_state.documents)} documents")
    
    # Option to view documents
    if st.sidebar.checkbox("View Uploaded Documents"):
        for doc in st.session_state.documents:
            with st.sidebar.expander(f"{doc['name']}"):
                st.write(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Calculator function
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

# Simple document-based Q&A system
def document_qa(query):
    """Answer questions based on uploaded documents and predefined facts"""
    query_lower = query.lower()
    
    # First check user-uploaded documents
    if st.session_state.documents:
        # Very simple retrieval: check if query terms appear in documents
        relevant_passages = []
        
        # Split query into keywords (remove common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'what', 'how', 'where', 'when', 'who', 'why'}
        keywords = [word for word in query_lower.split() if word not in stop_words]
        
        for doc in st.session_state.documents:
            content_lower = doc['content'].lower()
            
            # Check if any keyword appears in the content
            if any(keyword in content_lower for keyword in keywords):
                # Find the most relevant paragraph
                paragraphs = [p for p in content_lower.split('\n\n') if p.strip()]
                
                for paragraph in paragraphs:
                    if any(keyword in paragraph for keyword in keywords):
                        source = f"(From {doc['name']})"
                        relevant_passages.append(f"{paragraph} {source}")
                        break
        
        if relevant_passages:
            # Return the most relevant passage
            return relevant_passages[0]
    
    # Sample facts as fallback
    facts = {
        "rag": "RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by retrieving relevant information from external sources.",
        "assistant": "The RAG-Powered Assistant uses retrieval-augmented generation with a multi-agent approach to provide accurate answers.",
        "system requirements": "The system requires Python 3.8 or higher, 8GB RAM, and an internet connection for LLM API access.",
        "file formats": "The system supports TXT, MD, and PDF file formats for document ingestion.",
        "pricing": "The basic version is free with limited document processing. Premium plans start at $19.99/month.",
        "document upload": "To upload documents, navigate to the Document Management section and select files to upload.",
        "tools": "The system includes specialized tools like Calculator and Dictionary, plus the RAG pipeline."
    }
    
    # Check if any keywords match
    for keyword, answer in facts.items():
        if keyword in query_lower:
            return answer
    
    # Default response
    if st.session_state.documents:
        return "I couldn't find relevant information in your documents. Could you try rephrasing your question?"
    else:
        return "I don't have specific information about that in my knowledge base. You can also upload documents using the sidebar to ask questions about them."

# Function to determine which tool to use
def route_query(query):
    """Determine which tool should handle the query"""
    query_lower = query.lower()
    
    # Check for calculation keywords
    calc_keywords = ['calculate', 'compute', 'what is', 'square root', 'plus', 'minus', 'times', 'divided']
    for keyword in calc_keywords:
        if keyword in query_lower and any(char.isdigit() for char in query_lower):
            return "calculator"
    
    # All other queries go to document QA
    return "document_qa"

# Process user input
prompt = st.chat_input("What would you like to know?")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Determine which tool to use
    tool = route_query(prompt)
    
    # Process with appropriate tool
    if tool == "calculator":
        response = calculate(prompt)
    else:
        response = document_qa(prompt)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response)

# Footer with instructions
st.markdown("---")
st.markdown("""
## How to use this system:

1. **Upload your own documents** using the sidebar to ask questions about them

2. **Ask calculation questions** like:
   - "What is the square root of 88?"
   - "Calculate 15 + 27"

3. **Ask about the RAG system** like:
   - "What is RAG?"
   - "What are the system requirements?"
   - "What file formats are supported?"
   - "How much does it cost?"
   - "What tools are available?"

4. **Ask questions about your uploaded documents**
   - Questions will be matched to relevant content in your documents
   - Results include the source document name
""")