"""
Generator module for RAG (Retrieval-Augmented Generation) system.

This module provides functionality for generating responses based on retrieved documents
and user queries. It includes a simple 'mock'generator implementation that can be extended
or replaced with more sophisticated language models.
"""

import logging

from rag.llm import OpenAI_LLM
from schema.document import Document
from schema.generator_config import GeneratorConfig

logger = logging.getLogger(__name__)

PROMPT_TEMPLATES = {
    "loose": "Answer the query based on the following documents:",
    "strict": """Answer the query based on the following documents.  
    If the query is not supported by the documents, return 'I don't know.'
    Do not include any information in your response that is not included in the
    attached documents.""",
}

class Generator:
    """
    A simple generator for creating responses based on retrieved documents.
    
    This class provides a basic implementation that can be extended with more
    sophisticated language models. It maintains the last generated prompt for
    debugging and analysis purposes.
    
    Attributes:
        last_prompt (str): The most recently generated prompt for debugging.
    """
    
    def __init__(self, config: GeneratorConfig):
        """
        Initialize the Generator with an empty last prompt.
        """
        self.last_prompt = ""
        self.config = config
        self.llm = OpenAI_LLM()

    def generate(self, query: str, documents: list[Document])-> str:
        """
        Generate a response based on the query and retrieved documents.
        
        This method creates a prompt combining the query with the retrieved
        documents and returns a response. Currently returns the first document's
        content as a placeholder implementation.
        
        Args:
            query (str): The user's query to answer.
            documents (list[Document]): List of retrieved documents to use for generation.
            mode (str): The mode to use for generation.  Defaults to "loose".
        
        Returns:
            str: The generated response based on the documents and query.
        """
        logger.info(f"Generating response for query: {query} with mode: {self.config.mode}")
        llm_boilerplate = PROMPT_TEMPLATES.get(self.config.mode, PROMPT_TEMPLATES['loose'])
        documents_str = "\n".join([doc.data for doc in documents])
        self.last_prompt= f"{llm_boilerplate}\n\n{documents_str}\n\nQuery: {query}"
        return self.llm.generate_response(self.last_prompt, "gpt-4o-mini")
    
    def get_last_prompt(self)-> str:
        """
        Get the last generated prompt for debugging purposes.
        
        Returns:
            str: The most recently generated prompt.
        """
        return self.last_prompt
