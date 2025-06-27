"""
Embedding module for RAG (Retrieval-Augmented Generation) system.

This module provides functionality for generating text embeddings using pre-trained
sentence transformer models. It supports both single text and batch text embedding
operations.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    A class for generating text embeddings using sentence transformer models.
    
    This class wraps the SentenceTransformer library to provide a simple interface
    for converting text into vector representations that can be used for semantic
    search and similarity comparisons in RAG systems.
    
    Attributes:
        model_name (str): The name of the pre-trained sentence transformer model
        model (SentenceTransformer): The loaded sentence transformer model instance
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Embedder with a specified sentence transformer model.
        Default is 'all-MiniLM-L6-v2' which is a good balance of performance and speed fr
        running tests locally.
        
        Args:
            model_name (str): The name of the pre-trained model to use.
                             Defaults to 'all-MiniLM-L6-v2' which is a good
                             balance of performance and speed.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _embed(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Internal method to generate embeddings for text input.
        
        This method handles the core embedding logic and can process either
        a single string or a list of strings.
        
        Args:
            input (Union[str, List[str]]): Text to embed. Can be a single string
                                          or a list of strings for batch processing.
        
        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s). Returns a
                                                  single vector for string input,
                                                  or a list of vectors for list input.
        
        Raises:
            ValueError: If no text is provided for embedding.
        """
        if not input:
            raise ValueError("No text provided for embedding")
        return self.model.encode(input).tolist()

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text (str): The text string to embed.
        
        Returns:
            List[float]: A vector representation of the input text.
        """
        return self._embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of text strings.
        
        This method is more efficient than calling embed() multiple times
        as it processes all texts in a single forward pass.
        
        Args:
            texts (List[str]): List of text strings to embed.
        
        Returns:
            List[List[float]]: List of vector representations, one for each input text.
        """
        return self._embed(texts)


class ChromaEmbedder:
    def __init__(self, embedder: 'Embedder'):
        self.embedder = embedder
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embedder.embed_batch(input)
    

    