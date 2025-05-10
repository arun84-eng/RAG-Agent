import os
from typing import List, Dict, Any
from openai import OpenAI

# The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# Do not change this unless explicitly requested by the user
OPENAI_MODEL = "gpt-4o"

class LLMService:
    def __init__(self):
        """Initialize the LLM service with OpenAI"""
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
    
    def generate_answer(self, query: str, context: List[str] = None) -> str:
        """
        Generate an answer based on the query and optional context
        
        Args:
            query: User's question
            context: List of relevant document chunks (optional)
            
        Returns:
            Generated answer
        """
        # Prepare the prompt with context if available
        if context and len(context) > 0:
            context_text = "\n\n".join(context)
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
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that gives accurate, concise answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more factual responses
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()

    def detect_tool_usage(self, query: str) -> str:
        """
        Detect if the query should be routed to a specialized tool
        
        Args:
            query: User's question
            
        Returns:
            Tool name or "rag" for default RAG pipeline
        """
        prompt = f"""Analyze the following query and determine if it should be processed by a specialized tool or if it should use a standard RAG pipeline.

Query: {query}

If the query is asking for a calculation (like arithmetic, square root, percentages, etc.), respond with only "calculator".
If the query is asking for a definition (keywords like "define", "what is", "meaning of"), respond with only "dictionary".
Otherwise, respond with only "rag".

Your response (just one word):"""

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
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
