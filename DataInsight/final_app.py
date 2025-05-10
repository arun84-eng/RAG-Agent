import os
import streamlit as st
import math
import re
import tempfile
from typing import List, Dict, Any

# Set page configuration
st.set_page_config(page_title="RAG-Powered Q&A Assistant", page_icon="🤖")
st.title("RAG-Powered Multi-Agent Q&A Assistant")

# --- DOCUMENT PROCESSING ---

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks of approximately the specified size."""
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
        else:
            # Include the punctuation and space
            end = last_sentence_break + 2
        
        # Add the chunk
        chunks.append(text[start:end])
        
        # Move to the next chunk, accounting for overlap
        start = end - chunk_overlap
    
    return chunks

def process_document(file_content, file_name):
    """Process a document file, chunking it and preparing for search."""
    try:
        # Chunk the document
        text_chunks = chunk_text(file_content)
        
        # Create document chunks with metadata
        document_chunks = []
        for i, chunk in enumerate(text_chunks):
            document_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": file_name,
                    "chunk_id": i,
                    "total_chunks": len(text_chunks)
                }
            })
        
        return document_chunks
    
    except Exception as e:
        st.error(f"Error processing document {file_name}: {str(e)}")
        return []

def create_sample_documents():
    """Create sample documents with example content"""
    sample_docs = {}
    
    # Sample company FAQ
    sample_docs["company_faq.txt"] = """
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
    sample_docs["product_specs.txt"] = """
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
    sample_docs["user_manual.txt"] = """
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
    
    return sample_docs

# --- CALCULATOR TOOL ---

class Calculator:
    """Tool for performing calculations based on user queries"""
    
    def __init__(self):
        """Initialize the calculator tool"""
        pass
    
    def _extract_math_expression(self, query: str) -> str:
        """Extract a mathematical expression from a natural language query"""
        # Remove common phrases
        cleaned_query = query.lower()
        cleaned_query = re.sub(r'calculate\s+', '', cleaned_query)
        cleaned_query = re.sub(r'what\s+is\s+', '', cleaned_query)
        cleaned_query = re.sub(r'the\s+', '', cleaned_query)
        
        # Replace common mathematical terms with symbols
        cleaned_query = re.sub(r'\s*squared\s*', '** 2', cleaned_query)
        cleaned_query = re.sub(r'\s*cubed\s*', '** 3', cleaned_query)
        cleaned_query = re.sub(r'square\s+root\s+of\s+', 'math.sqrt(', cleaned_query)
        if 'square root' in cleaned_query and ')' not in cleaned_query:
            cleaned_query += ')'
        
        # Replace percentage calculations
        percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)', cleaned_query)
        if percentage_match:
            a, b = percentage_match.groups()
            cleaned_query = f"({a} / 100) * {b}"
        
        # Replace x with *
        cleaned_query = re.sub(r'(\d+)\s*x\s*(\d+)', r'\1 * \2', cleaned_query)
        
        # Clean up the expression
        cleaned_query = re.sub(r'[^\d\s\+\-\*\/\(\)\.\^\%]+', '', cleaned_query)
        cleaned_query = cleaned_query.strip()
        
        return cleaned_query
    
    def _safe_eval(self, expression: str):
        """Safely evaluate a mathematical expression"""
        # Define a limited set of allowed names
        allowed_names = {
            'math': math,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp
        }
        
        # Evaluate the expression in a restricted environment
        try:
            return eval(expression, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            raise ValueError(f"Failed to evaluate the expression: {e}")
    
    def run(self, query: str) -> Dict[str, Any]:
        """Perform a calculation based on the query"""
        try:
            # Extract the mathematical expression
            expression = self._extract_math_expression(query)
            
            # Calculate the result
            result = self._safe_eval(expression)
            
            # Format the result for better readability
            if isinstance(result, float) and result.is_integer():
                formatted_result = int(result)
            else:
                formatted_result = result
            
            return {
                "tool": "calculator",
                "result": formatted_result,
                "expression": expression,
                "answer": f"The calculation result is: {formatted_result}",
                "context": []
            }
        except Exception as e:
            return {
                "tool": "calculator",
                "error": str(e),
                "answer": f"I couldn't perform this calculation. Error: {str(e)}",
                "context": []
            }

# --- DICTIONARY TOOL ---

class Dictionary:
    """Tool for providing definitions of terms"""
    
    def __init__(self):
        """Initialize the dictionary tool"""
        self.definitions = {
            "rag": "RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by retrieving relevant information from external sources before generating answers.",
            "llm": "LLM (Large Language Model) is an AI system trained on vast amounts of text data to understand and generate human-like text.",
            "vector embedding": "A vector embedding is a numerical representation of text that captures its semantic meaning in a high-dimensional space, enabling similarity comparisons.",
            "retrieval": "In the context of RAG systems, retrieval refers to the process of finding relevant information from a knowledge base in response to a query.",
            "generation": "In AI systems, generation refers to the process of creating new content, such as text, based on learned patterns and provided inputs.",
            "chunk": "In document processing, a chunk is a segment of text created by splitting larger documents into smaller, manageable pieces for more effective retrieval.",
            "api": "API (Application Programming Interface) is a set of rules that allow different software applications to communicate with each other.",
            "neural network": "A neural network is a computational model inspired by the human brain, consisting of interconnected nodes (neurons) that process information.",
            "machine learning": "Machine learning is a field of AI that enables systems to learn and improve from experience without being explicitly programmed.",
            "artificial intelligence": "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems."
        }
    
    def _extract_term(self, query: str) -> str:
        """Extract the term to be defined from a natural language query"""
        # Remove common phrase patterns
        cleaned_query = query.lower()
        cleaned_query = re.sub(r'define\s+', '', cleaned_query)
        cleaned_query = re.sub(r'what\s+is\s+', '', cleaned_query)
        cleaned_query = re.sub(r'what\s+are\s+', '', cleaned_query)
        cleaned_query = re.sub(r'the\s+meaning\s+of\s+', '', cleaned_query)
        cleaned_query = re.sub(r'meaning\s+of\s+', '', cleaned_query)
        
        # Remove punctuation at the end
        cleaned_query = re.sub(r'[\?\.\!]+$', '', cleaned_query)
        
        return cleaned_query.strip()
    
    def run(self, query: str) -> Dict[str, Any]:
        """Provide definition for a term in the query"""
        try:
            # Extract the term to define
            term = self._extract_term(query)
            
            # Try to find the term in our dictionary
            found = False
            definition = ""
            
            # Check for exact match
            if term in self.definitions:
                definition = self.definitions[term]
                found = True
            
            # Check for partial match
            if not found:
                for key, value in self.definitions.items():
                    if key in term or term in key:
                        definition = f"{key.capitalize()}: {value}"
                        found = True
                        break
            
            if found:
                return {
                    "tool": "dictionary",
                    "term": term,
                    "answer": definition,
                    "context": []
                }
            else:
                return {
                    "tool": "dictionary",
                    "term": term,
                    "answer": f"I don't have a definition for '{term}' in my dictionary.",
                    "context": []
                }
        except Exception as e:
            return {
                "tool": "dictionary",
                "error": str(e),
                "answer": f"I couldn't retrieve a definition. Error: {str(e)}",
                "context": []
            }

# --- DOCUMENT SEARCH ---

class SimpleDocumentSearch:
    """
    Simple document search engine that matches queries to document content
    without relying on external vector embeddings.
    """
    
    def __init__(self):
        """Initialize the document search engine"""
        self.documents = []
        
    def add_documents(self, documents):
        """Add documents to the search engine"""
        self.documents.extend(documents)
    
    def search(self, query, top_k=3):
        """
        Search for documents relevant to the query using simple keyword matching
        """
        if not self.documents:
            return []
        
        query_keywords = self._extract_keywords(query)
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in self.documents:
            score = self._calculate_score(doc["text"], query_keywords)
            if score > 0:
                scored_docs.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score
                })
        
        # Sort by score (descending) and take top_k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]
    
    def _extract_keywords(self, text):
        """Extract keywords from text by removing stop words"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'of', 'to', 'in', 
                     'that', 'it', 'with', 'as', 'for', 'on', 'was', 'be', 'at', 'this', 
                     'by', 'i', 'you', 'we', 'they', 'what', 'how', 'why', 'when', 'where'}
        
        # Tokenize and filter
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _calculate_score(self, text, query_keywords):
        """Calculate a simple relevance score based on keyword matches"""
        if not query_keywords:
            return 0
        
        text_lower = text.lower()
        text_keywords = self._extract_keywords(text_lower)
        
        # Count matches
        matches = 0
        for qk in query_keywords:
            if qk in text_lower:
                matches += 1
            # Bonus points for exact keyword matches
            if qk in text_keywords:
                matches += 0.5
                
        # Normalize score by query length
        score = matches / len(query_keywords)
        
        return score

# --- QUERY PROCESSING ---

def detect_tool_usage(query: str) -> str:
    """Determine which tool to use for the query"""
    query_lower = query.lower()
    
    # Check for calculation patterns
    calc_patterns = [
        r'\bcalculate\b',
        r'what\s+is\s+\d+',
        r'what\s+is\s+the\s+square\s+root',
        r'\d+\s*[\+\-\*\/]\s*\d+',
        r'square\s+root\s+of\s+\d+'
    ]
    
    for pattern in calc_patterns:
        if re.search(pattern, query_lower):
            return "calculator"
    
    # Check for definition patterns
    define_patterns = [
        r'\bdefine\b',
        r'what\s+is\s+a\s+\w+',
        r'what\s+does\s+\w+\s+mean',
        r'meaning\s+of\s+\w+'
    ]
    
    for pattern in define_patterns:
        if re.search(pattern, query_lower):
            return "dictionary"
    
    # Default to RAG
    return "rag"

def process_query(query, search_engine, calculator, dictionary):
    """Process query using the appropriate tool"""
    # Detect which tool to use
    tool_type = detect_tool_usage(query)
    
    # Initialize result
    result = {
        "tool": tool_type,
        "answer": "",
        "context": []
    }
    
    # Route to the appropriate tool
    if tool_type == "calculator":
        calc_result = calculator.run(query)
        return calc_result
    
    elif tool_type == "dictionary":
        dict_result = dictionary.run(query)
        return dict_result
    
    else:  # Default to document search
        # Search for relevant documents
        search_results = search_engine.search(query, top_k=3)
        
        if search_results:
            # Extract texts from results
            context_texts = [result["text"] for result in search_results]
            
            # Format a response that combines the retrieved information
            context_with_sources = []
            for result in search_results:
                source = result["metadata"]["source"]
                context_with_sources.append(f"From {source}: {result['text']}")
            
            answer = "Based on the documents, I found the following information:\n\n"
            answer += "\n\n".join(context_with_sources)
            
            result = {
                "tool": "rag",
                "answer": answer,
                "context": context_texts
            }
        else:
            result = {
                "tool": "rag",
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "context": []
            }
    
    return result

# --- MAIN APPLICATION ---

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.documents = []
    st.session_state.search_engine = None

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
    st.session_state.documents = []
    for file in uploaded_files:
        try:
            content = file.getvalue().decode('utf-8')
            chunks = process_document(content, file.name)
            st.session_state.documents.extend(chunks)
            st.success(f"Successfully processed: {file.name}")
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

# Load sample documents if selected
if use_sample_docs:
    sample_docs = create_sample_documents()
    
    sample_chunks = []
    for doc_name, doc_content in sample_docs.items():
        # Process the sample document content
        chunks = process_document(doc_content, doc_name)
        sample_chunks.extend(chunks)
    
    if sample_chunks:
        if st.session_state.documents:
            st.session_state.documents.extend(sample_chunks)
        else:
            st.session_state.documents = sample_chunks
        st.info(f"Loaded {len(sample_chunks)} chunks from sample documents")

# Initialize search engine and tools
if st.session_state.documents and (not st.session_state.initialized or st.button("Reinitialize System")):
    with st.spinner("Creating search engine..."):
        st.session_state.search_engine = SimpleDocumentSearch()
        st.session_state.search_engine.add_documents(st.session_state.documents)
        st.session_state.calculator = Calculator()
        st.session_state.dictionary = Dictionary()
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
            # Process query with the appropriate tool
            result = process_query(
                user_query,
                st.session_state.search_engine,
                st.session_state.calculator,
                st.session_state.dictionary
            )
            
            # Display the agent's decision process
            st.subheader("Agent Workflow")
            
            st.markdown("**Tool Selected:**")
            st.info(result["tool"])
            
            # Display retrieved context if available
            if result["context"]:
                st.subheader("Retrieved Context")
                for i, context_item in enumerate(result["context"]):
                    with st.expander(f"Context {i+1}"):
                        st.write(context_item)
            
            # Display the final answer
            st.subheader("Answer")
            st.write(result["answer"])
            
            # Display additional tool-specific information
            if result["tool"] == "calculator" and "expression" in result:
                with st.expander("View Calculation Details"):
                    st.code(result["expression"])
                    
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Footer with instructions
st.markdown("---")
st.markdown("""
**Instructions:**
- Upload your own documents or use the sample documents
- Click 'Reinitialize System' after document selection
- Ask questions about the content of the documents
- Try different query types:
  - "What are the main features of the product?"
  - "Calculate the square root of 88"
  - "Define RAG"
""")