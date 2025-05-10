import os
import logging
from typing import Dict, Any, List

from tools import Calculator, Dictionary
from llm_integration import LLMService
from vector_store import FAISSVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAgent:
    """Simple agent that routes queries to appropriate tools"""
    
    def __init__(self, vector_store: FAISSVectorStore):
        """Initialize the agent with tools"""
        self.vector_store = vector_store
        self.llm_service = LLMService()
        self.calculator = Calculator()
        self.dictionary = Dictionary()
        self.log = []
        
    def add_to_log(self, message: str):
        """Add a message to the log"""
        self.log.append(message)
        
    def get_log(self):
        """Get the complete log"""
        return "\n".join(self.log)
        
    def clear_log(self):
        """Clear the log"""
        self.log = []
        
    def rag_tool(self, query: str) -> Dict[str, Any]:
        """
        RAG pipeline tool that retrieves relevant context and generates an answer
        
        Args:
            query: User's question
            
        Returns:
            Result with answer and retrieved context
        """
        self.add_to_log(f"Using RAG tool for query: {query}")
        
        # Search for relevant document chunks
        search_results = self.vector_store.search(query, top_k=3)
        self.add_to_log(f"Retrieved {len(search_results)} document chunks")
        
        # Extract text from results
        context_texts = [result["text"] for result in search_results]
        
        # Generate answer using the LLM
        answer = self.llm_service.generate_answer(query, context_texts)
        self.add_to_log(f"Generated answer using LLM")
        
        return {
            "answer": answer,
            "context": context_texts,
            "search_results": search_results
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query by routing to the appropriate tool
        
        Args:
            query: User's question
            
        Returns:
            Result with the answer and processing information
        """
        self.clear_log()
        self.add_to_log(f"Processing query: {query}")
        
        # Detect which tool to use
        tool_type = self.llm_service.detect_tool_usage(query)
        self.add_to_log(f"Detected tool type: {tool_type}")
        
        try:
            result = {
                "tool": tool_type,
                "chain": "Query → Tool → Answer",
                "context": []
            }
            
            # Route to the appropriate tool
            if tool_type == "calculator":
                self.add_to_log("Routing to Calculator tool")
                calc_result = self.calculator.run(query)
                result["answer"] = calc_result["answer"]
                
            elif tool_type == "dictionary":
                self.add_to_log("Routing to Dictionary tool")
                dict_result = self.dictionary.run(query)
                result["answer"] = dict_result["answer"]
                
            else:  # Default to RAG
                self.add_to_log("Routing to RAG pipeline")
                rag_result = self.rag_tool(query)
                result["answer"] = rag_result["answer"]
                result["context"] = rag_result["context"]
            
            # Add the log
            result["log"] = self.get_log()
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            self.add_to_log(f"Error: {str(e)}")
            
            return {
                "answer": f"Sorry, I encountered an error processing your request: {str(e)}",
                "tool": tool_type,
                "chain": "Error in processing",
                "context": [],
                "log": self.get_log()
            }

def create_agent(vector_store: FAISSVectorStore) -> SimpleAgent:
    """
    Create a simple agent for routing queries
    
    Args:
        vector_store: Vector store for document retrieval
        
    Returns:
        Initialized agent
    """
    return SimpleAgent(vector_store)

def query_agent(agent: SimpleAgent, query: str) -> Dict[str, Any]:
    """
    Process a user query through the agent
    
    Args:
        agent: Initialized agent
        query: User's question
        
    Returns:
        Result containing the answer, tool used, and processing chain
    """
    return agent.process_query(query)
