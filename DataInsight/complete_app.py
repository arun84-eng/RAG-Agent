import os
import streamlit as st
import math
import re
import requests
import tempfile
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI

# Set page configuration
st.set_page_config(page_title="RAG-Powered Q&A Assistant", page_icon="🤖")
st.title("RAG-Powered Multi-Agent Q&A Assistant")

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# ---- DOCUMENT PROCESSING ----

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

def process_document(file):
    """Process a document file, chunking it and preparing for vector store."""
    try:
        # Read file content
        content = file.getvalue().decode('utf-8')
        
        # Get basic document metadata
        doc_name = file.name
        
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
        st.error(f"Error processing document {file.name}: {str(e)}")
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

# ---- VECTOR STORE ----

class SimpleVectorStore:
    """Simple vector store that uses OpenAI embeddings"""
    
    def __init__(self):
        """Initialize the vector store"""
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        if not documents:
            return
        
        # Store documents
        self.documents.extend(documents)
        
        # Generate and store embeddings for each document
        for doc in documents:
            self.embeddings.append(self._get_embedding(doc["text"]))
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embeddings for text using OpenAI"""
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error getting embedding: {e}")
            # Return a zero vector if there's an error
            return [0.0] * 1536
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between vectors"""
        # Convert to numpy arrays if they aren't already
        a = np.array(a)
        b = np.array(b)
        
        # Compute cosine similarity
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for documents similar to the query"""
        if not self.documents:
            return []
        
        # Get embedding for the query
        query_embedding = self._get_embedding(query)
        
        # Calculate similarity scores
        scores = []
        for doc_embedding in self.embeddings:
            score = self._cosine_similarity(query_embedding, doc_embedding)
            scores.append(score)
        
        # Get the indices of the top_k highest scores
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            result = {
                "text": self.documents[idx]["text"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(scores[idx])
            }
            results.append(result)
        
        return results

# ---- SPECIALIZED TOOLS ----

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
                "result": formatted_result,
                "expression": expression,
                "answer": f"The calculation result is: {formatted_result}"
            }
        except Exception as e:
            return {
                "error": str(e),
                "answer": f"I couldn't perform this calculation. Error: {str(e)}"
            }

class Dictionary:
    """Tool for providing definitions of terms"""
    
    def __init__(self):
        """Initialize the dictionary tool"""
        pass
    
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
    
    def _get_definition_from_api(self, term: str) -> List[Dict[str, str]]:
        """Get definition from a dictionary API"""
        # Use Free Dictionary API
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Extract definitions
                definitions = []
                for entry in data:
                    for meaning in entry.get('meanings', []):
                        part_of_speech = meaning.get('partOfSpeech', '')
                        for definition in meaning.get('definitions', []):
                            definitions.append({
                                'definition': definition.get('definition', ''),
                                'part_of_speech': part_of_speech
                            })
                
                return definitions[:3]  # Return top 3 definitions
            else:
                return []
        except Exception:
            return []
    
    def run(self, query: str) -> Dict[str, Any]:
        """Provide definition for a term in the query"""
        try:
            # Extract the term to define
            term = self._extract_term(query)
            
            # Get definitions from API
            definitions = self._get_definition_from_api(term)
            
            if definitions:
                # Format the definitions
                formatted_definitions = []
                for i, def_item in enumerate(definitions):
                    formatted_definitions.append(
                        f"{i+1}. ({def_item['part_of_speech']}) {def_item['definition']}"
                    )
                
                definition_text = "\n".join(formatted_definitions)
                answer = f"Definition of '{term}':\n{definition_text}"
                
                return {
                    "term": term,
                    "definitions": definitions,
                    "answer": answer
                }
            else:
                # Fallback to a more general response
                return {
                    "term": term,
                    "definitions": [],
                    "answer": f"I couldn't find a specific definition for '{term}'. This term might be specialized, technical, or not in common dictionaries."
                }
        except Exception as e:
            return {
                "error": str(e),
                "answer": f"I couldn't retrieve a definition. Error: {str(e)}"
            }

# ---- LLM INTEGRATION ----

def detect_tool_usage(query: str) -> str:
    """Detect if the query should be routed to a specialized tool"""
    prompt = f"""Analyze the following query and determine if it should be processed by a specialized tool or if it should use a standard RAG pipeline.

Query: {query}

If the query is asking for a calculation (like arithmetic, square root, percentages, etc.), respond with only "calculator".
If the query is asking for a definition (keywords like "define", "what is", "meaning of"), respond with only "dictionary".
Otherwise, respond with only "rag".

Your response (just one word):"""

    response = client.chat.completions.create(
        model="gpt-4o",  # The newest OpenAI model
        messages=[
            {"role": "system", "content": "You are a query classifier that categorizes questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Zero temperature for deterministic outputs
        max_tokens=10
    )
    
    tool = response.choices[0].message.content.strip().lower()
    
    # Validate the response to ensure it's one of the expected tools
    if tool not in ["calculator", "dictionary", "rag"]:
        return "rag"  # Default to RAG if classification is uncertain
    
    return tool

def generate_rag_answer(query: str, context_texts: List[str]) -> str:
    """Generate an answer based on the query and context"""
    # Prepare the prompt with context
    if context_texts and len(context_texts) > 0:
        context_text = "\n\n".join(context_texts)
        prompt = f"""Answer the question based on the following context. If the question cannot be answered using the information provided, say "I don't have enough information to answer this question." Don't try to make up an answer.

Context:
{context_text}

Question: {query}

Answer:"""
    else:
        prompt = f"""Answer the question based on your knowledge. If you're not sure, please say so.

Question: {query}

Answer:"""

    # Call the LLM
    response = client.chat.completions.create(
        model="gpt-4o",  # The newest OpenAI model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that gives accurate, concise answers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # Lower temperature for more factual responses
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()

# ---- MAIN APPLICATION ----

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.documents = []
    st.session_state.vector_store = None

# Document upload section
st.header("Document Management")

# File uploader for custom documents
uploaded_files = st.file_uploader("Upload your own documents (optional)", 
                                 accept_multiple_files=True, 
                                 type=['txt', 'md'])

# Sample docs selection
use_sample_docs = st.checkbox("Use sample documents", value=True)

# Process document uploads
if uploaded_files:
    st.session_state.documents = []
    for file in uploaded_files:
        # Process the document
        chunks = process_document(file)
        st.session_state.documents.extend(chunks)
        st.success(f"Successfully processed: {file.name}")

# Load sample documents if selected
if use_sample_docs:
    sample_docs = create_sample_documents()
    
    sample_chunks = []
    for doc_name, doc_content in sample_docs.items():
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp_file:
            temp_file.write(doc_content)
            temp_path = temp_file.name
        
        # Open and process the file
        with open(temp_path, 'r') as f:
            content = f.read()
        
        # Chunk the document
        text_chunks = chunk_text(content)
        
        # Create document chunks with metadata
        for i, chunk in enumerate(text_chunks):
            sample_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": doc_name,
                    "chunk_id": i,
                    "total_chunks": len(text_chunks)
                }
            })
        
        # Remove the temporary file
        os.unlink(temp_path)
    
    if sample_chunks:
        if st.session_state.documents:
            st.session_state.documents.extend(sample_chunks)
        else:
            st.session_state.documents = sample_chunks
        st.info(f"Loaded {len(sample_chunks)} chunks from sample documents")

# Initialize vector store and tools
if st.session_state.documents and (not st.session_state.initialized or st.button("Reinitialize System")):
    with st.spinner("Creating vector store..."):
        st.session_state.vector_store = SimpleVectorStore()
        st.session_state.vector_store.add_documents(st.session_state.documents)
    
    st.session_state.initialized = True
    st.session_state.calculator = Calculator()
    st.session_state.dictionary = Dictionary()
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
            # Detect which tool to use
            tool_type = detect_tool_usage(user_query)
            
            # Initialize result dictionary
            result = {
                "tool": tool_type,
                "context": [],
                "answer": ""
            }
            
            # Route to the appropriate tool
            if tool_type == "calculator":
                calc_result = st.session_state.calculator.run(user_query)
                result["answer"] = calc_result["answer"]
                result["expression"] = calc_result.get("expression", "")
                
            elif tool_type == "dictionary":
                dict_result = st.session_state.dictionary.run(user_query)
                result["answer"] = dict_result["answer"]
                result["term"] = dict_result.get("term", "")
                
            else:  # Default to RAG
                # Search for relevant document chunks
                search_results = st.session_state.vector_store.search(user_query, top_k=3)
                
                # Extract text from results
                context_texts = [result["text"] for result in search_results]
                
                # Generate answer using the LLM
                answer = generate_rag_answer(user_query, context_texts)
                
                result["answer"] = answer
                result["context"] = context_texts
                result["search_results"] = search_results
            
            # Display the agent's decision process
            st.subheader("Agent Workflow")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Tool Selected:**")
                st.info(result["tool"])
            
            # Display retrieved context if RAG was used
            if result["context"]:
                st.subheader("Retrieved Context")
                for i, context_item in enumerate(result["context"]):
                    with st.expander(f"Context {i+1}"):
                        st.write(context_item)
            
            # Display the final answer
            st.subheader("Answer")
            st.write(result["answer"])
            
            # Display additional tool-specific information
            if tool_type == "calculator" and "expression" in result:
                with st.expander("View Calculation Details"):
                    st.code(result["expression"])
                    
            if tool_type == "dictionary" and "term" in result:
                with st.expander("View Term Details"):
                    st.write(f"Searched for: '{result['term']}'")
                
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