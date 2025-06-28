"""
Pipeline module for RAG (Retrieval-Augmented Generation) system.

This module provides the main pipeline that orchestrates the retrieval and generation
components of the RAG system. It provides a simple interface for running end-to-end
RAG queries.
"""

from rag.retriever import Retriever
from rag.generator import Generator


class RagPipeline:
    """
    Main pipeline for RAG (Retrieval-Augmented Generation) system.
    
    This class orchestrates the retrieval and generation components to provide
    a complete RAG solution. It takes a query, retrieves relevant documents,
    and generates a response based on those documents.
    
    Attributes:
        retriever (Retriever): The document retriever component.
        generator (Generator): The response generator component.
    """
    
    def __init__(self, retriever: Retriever, generator: Generator):
        """
        Initialize the RAG pipeline with retriever and generator components.
        
        Args:
            retriever (Retriever): The document retriever to use for finding relevant documents.
            generator (Generator): The response generator to use for creating answers.
        """
        self.generator = generator
        self.retriever = retriever

    def run(self, query: str):
        """
        Run the complete RAG pipeline on a given query.
        
        This method performs the full RAG workflow: retrieving relevant documents
        based on the query and then generating a response using those documents.
        
        Args:
            query (str): The user's query to process.
        
        Returns:
            str: The generated response based on retrieved documents.
        """
        documents = self.retriever.retrieve(query)
        return self.generator.generate(query, documents)





