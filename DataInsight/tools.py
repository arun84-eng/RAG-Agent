import math
import re
import requests
from typing import Dict, Any, Union, List

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
    
    def _safe_eval(self, expression: str) -> Union[float, int]:
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

class Dictionary:
    """Tool for providing definitions of terms"""
    
    def __init__(self):
        """Initialize the dictionary tool"""
        pass
    
    def _extract_term(self, query: str) -> str:
        """
        Extract the term to be defined from a natural language query
        
        Args:
            query: User's question asking for a definition
            
        Returns:
            Extracted term
        """
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
        """
        Get definition from a dictionary API
        
        Args:
            term: Term to define
            
        Returns:
            List of definitions with parts of speech
        """
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
        """
        Provide definition for a term in the query
        
        Args:
            query: User's question asking for a definition
            
        Returns:
            Dictionary with definitions and formatted answer
        """
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
