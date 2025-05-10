import os
import streamlit as st
import math
import re
from typing import Dict, Any

# Simple calculator implementation
class Calculator:
    """Tool for performing calculations based on user queries"""
    
    def __init__(self):
        """Initialize the calculator tool"""
        pass
    
    def _extract_math_expression(self, query: str) -> str:
        """
        Extract a mathematical expression from a natural language query
        
        Args:
            query: User's question containing a math problem
            
        Returns:
            Extracted mathematical expression
        """
        # Remove the "calculate" keyword and other common phrases
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
        """
        Safely evaluate a mathematical expression
        
        Args:
            expression: Math expression as a string
            
        Returns:
            Calculated result
        """
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
        """
        Perform a calculation based on the query
        
        Args:
            query: User's question containing a math problem
            
        Returns:
            Dictionary with result and explanation
        """
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

# Create Streamlit UI
st.set_page_config(page_title="Simple Calculator App", page_icon="🧮")

st.title("Simple Calculator")
st.markdown("Ask a calculation question like 'what is the square root of 88'")

# Initialize calculator
calculator = Calculator()

# Query input
user_query = st.text_input("Enter your calculation question:", placeholder="e.g., calculate the square root of 88")

# Process the query
if user_query:
    with st.spinner("Calculating..."):
        result = calculator.run(user_query)
        
        # Display the result
        st.subheader("Result")
        st.write(result["answer"])
        
        # Show the extracted expression
        with st.expander("See extracted expression"):
            st.code(result["expression"])