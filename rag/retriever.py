"""
Retriever module for RAG (Retrieval-Augmented Generation) system.

This module provides functionality for retrieving relevant documents based on user queries.
It combines semantic search using embeddings with re-ranking using cross-encoders to
improve retrieval quality.
"""

from sentence_transformers import CrossEncoder

from rag.embedding import Embedder
from rag.vectorstore import VectorStore
from schema.document import Document
import logging


logger = logging.getLogger(__name__)

class Retriever:
    """
    Document retriever that combines semantic search with re-ranking.
    
    This class provides a two-stage retrieval process: first using semantic search
    to find candidate documents, then re-ranking them using a cross-encoder model
    for improved relevance.
    
    Attributes:
        embedder (Embedder): The abstraction of the embedding model for semantic search.
        document_ranker (CrossEncoder): The cross-encoder model for re-ranking.
        vector_store (VectorStore): The vector database for document storage and retrieval.
    """
    
    def __init__(self, 
                 embedder_model_name: str = 'all-MiniLM-L6-v2',
                 ranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the Retriever with embedding and ranking models.
        
        Args:
            embedder_model_name (str): Name of the sentence transformer model for embeddings.
                                      Defaults to 'all-MiniLM-L6-v2'.
            ranker_model_name (str): Name of the cross-encoder model for re-ranking.
                                    Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        """
        self.embedder = Embedder(embedder_model_name)
        self.document_ranker = CrossEncoder(ranker_model_name)
        self.vector_store = VectorStore()
        self.last_documents = []
        logger.info(f"Retriever initialized with embedder: {embedder_model_name} and ranker: {ranker_model_name}")

    def retrieve(self, query: str, n_results: int = 10) -> list[Document]:
        """
        Retrieve and re-rank documents based on the query.
        
        This method performs a two-stage retrieval: first retrieving candidate
        documents using semantic search, then re-ranking them using a cross-encoder.
        
        Args:
            query (str): The search query.
            n_results (int): Number of documents to retrieve. Defaults to 10.
        
        Returns:
            list[Document]: List of retrieved documents, sorted by relevance score.
        """
        documents = self.vector_store.query(query, n_results)
        self.last_documents = documents
        return self.reorder_documents(documents, query)

    def reorder_documents(self, documents: list[Document], query: str) -> list[Document]:
        """
        Re-rank documents using a cross-encoder model.
        
        This method uses a cross-encoder to compute relevance scores between
        the query and each document, then sorts the documents by these scores.
        It also adds the rank to the document object.
        
        Args:
            documents (list[Document]): List of documents to re-rank.
            query (str): The query to rank documents against.
        
        Returns:
            list[Document]: List of documents sorted by relevance score (descending).
        """
        corpus = [doc.data for doc in documents]
        ranks = self.document_ranker.rank(query, corpus)
        for rank in ranks:
            documents[rank['corpus_id']].rank = rank['score']        
        return sorted(documents, key=lambda x: x.rank, reverse=True)
    
    def clear_last_documents(self):
        """
        Clear the last retrieved documents from memory.
        
        This method resets the last_documents list to an empty state. It's typically
        used for test isolation or when starting a new retrieval session to ensure
        no stale document references remain from previous queries.
        """
        self.last_documents = []

    