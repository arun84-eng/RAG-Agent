import os
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI

class FAISSVectorStore:
    def __init__(self):
        """Initialize a simple vector store that uses OpenAI embeddings"""
        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        
        # Store documents and their embeddings
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of document chunks with text and metadata
        """
        if not documents:
            return
        
        # Store documents with metadata
        self.documents.extend(documents)
        
        # Generate and store embeddings for each document
        for doc in documents:
            self.embeddings.append(self._get_embedding(doc["text"]))
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embeddings for text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
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
        """
        Search for documents similar to the query
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of document chunks with their similarity scores
        """
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

def create_vector_store(documents: List[Dict[str, Any]]) -> FAISSVectorStore:
    """
    Create and populate a vector store with documents
    
    Args:
        documents: List of document chunks with text and metadata
        
    Returns:
        Populated vector store
    """
    vector_store = FAISSVectorStore()
    vector_store.add_documents(documents)
    return vector_store
