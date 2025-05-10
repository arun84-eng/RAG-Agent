import os
import streamlit as st
import document_processing
import vector_store
import agent
import tempfile

st.set_page_config(page_title="RAG-Powered Q&A Assistant", page_icon="🤖")

st.title("RAG-Powered Multi-Agent Q&A Assistant")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.documents = []
    st.session_state.vector_store = None
    st.session_state.agent_executor = None

# Document upload section
st.header("Document Management")

# File uploader for custom documents
uploaded_files = st.file_uploader("Upload your own documents (optional)", 
                                 accept_multiple_files=True, 
                                 type=['txt', 'md', 'pdf'])

# Sample docs selection
use_sample_docs = st.checkbox("Use sample documents", value=True)

# Process document uploads
if uploaded_files:
    st.session_state.documents = []
    for file in uploaded_files:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name
        
        # Process the document
        try:
            chunks = document_processing.process_document(temp_path)
            st.session_state.documents.extend(chunks)
            st.success(f"Successfully processed: {file.name}")
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
        
        # Remove the temporary file
        os.unlink(temp_path)

# Load sample documents if selected
if use_sample_docs:
    sample_docs = [
        "sample_docs/company_faq.txt",
        "sample_docs/product_specs.txt", 
        "sample_docs/user_manual.txt"
    ]
    
    # Create sample docs if they don't exist
    if not os.path.exists("sample_docs"):
        os.makedirs("sample_docs")
        document_processing.create_sample_documents()
    
    sample_chunks = []
    for doc_path in sample_docs:
        if os.path.exists(doc_path):
            chunks = document_processing.process_document(doc_path)
            sample_chunks.extend(chunks)
    
    if sample_chunks:
        if st.session_state.documents:
            st.session_state.documents.extend(sample_chunks)
        else:
            st.session_state.documents = sample_chunks
        st.info(f"Loaded {len(sample_chunks)} chunks from sample documents")

# Initialize vector store and agent
if st.session_state.documents and (not st.session_state.initialized or st.button("Reinitialize System")):
    with st.spinner("Creating vector store..."):
        st.session_state.vector_store = vector_store.create_vector_store(st.session_state.documents)
    
    with st.spinner("Initializing agent..."):
        st.session_state.agent_executor = agent.create_agent(st.session_state.vector_store)
    
    st.session_state.initialized = True
    st.success("System initialized successfully!")

# Main Q&A Interface
st.header("Ask a Question")

# Show warning if system not initialized
if not st.session_state.initialized:
    st.warning("Please upload documents or use sample documents and initialize the system first.")

# Query input
user_query = st.text_input("Enter your question:", placeholder="e.g., What are the product features? or calculate 25^2")

# Process the query
if user_query and st.session_state.initialized:
    with st.spinner("Processing your question..."):
        try:
            result = agent.query_agent(st.session_state.agent_executor, user_query)
            
            # Display the agent's decision process
            st.subheader("Agent Workflow")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Tool Selected:**")
                st.info(result["tool"])
            
            with col2:
                st.markdown("**Processing Chain:**")
                st.info(result["chain"])
            
            # Display retrieved context if RAG was used
            if result["context"]:
                st.subheader("Retrieved Context")
                for i, context_item in enumerate(result["context"]):
                    with st.expander(f"Context {i+1}"):
                        st.write(context_item)
            
            # Display the final answer
            st.subheader("Answer")
            st.write(result["answer"])
            
            # Display full agent log for debugging
            with st.expander("View Detailed Agent Log"):
                st.code(result["log"])
                
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Footer with instructions
st.markdown("---")
st.markdown("""
**Instructions:**
- Upload your own documents or use the sample documents
- Ask questions about the content of the documents
- Try using keywords like 'calculate' or 'define' to trigger specialized tools
- Example questions:
  - "What are the main features of the product?"
  - "Calculate the square root of 25"
  - "Define artificial intelligence"
""")
