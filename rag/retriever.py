"""
Retriever module for RAG (Retrieval-Augmented Generation) system.

This module provides functionality for retrieving relevant documents based on user queries.
It combines semantic search using embeddings with re-ranking using cross-encoders to
improve retrieval quality.
"""

from sentence_transformers import CrossEncoder

from rag.embedding import Embedder
from rag.vectorstore import VectorStore
from schema.document import Document, MetaData
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

DEFAULT_DOCUMENT = Document(id='missing_document', 
                            metadata=MetaData(title="No documents retrieved for query", 
                                              source_species="",
                                              data_source="system"),
                            data="No documents retrieved for query",
                            rank=0)

INSUFFICIENT_RELEVANCE_DOCUMENT = Document(id='insufficient_relevance', 
                            metadata=MetaData(title="Document not relevant", 
                                              source_species="",
                                              data_source="system"),
                            data="Documents retrieved but not relevant to query",
                            rank=0)

logger = logging.getLogger(__name__)



class Retriever:
    """
    Document retriever that combines semantic search with re-ranking.
    
    This class provides a two-stage retrieval process: first using semantic search
    to find candidate documents, then re-ranking them using a cross-encoder model
    for improved relevance.

    Note - that during re-ranking, I use a sigmoid function to convert cross-encoder scores
    to a range of values from 0-1.  This permits me the flexibility to use a configurable
    re-ranking model and use a common threshold for all models.

    However, I am suspicious that the sigmoid function is not the best way to convert
    cross encoder scores for all models.  This is something to investigate as I proceed.
    
    Attributes:
        embedder (Embedder): The abstraction of the embedding model for semantic search.
        document_ranker (CrossEncoder): The cross-encoder model for re-ranking.
        vector_store (VectorStore): The vector database for document storage and retrieval.
    """
    
    def __init__(self, 
                 embedder_model_name: str = 'all-MiniLM-L6-v2',
                 ranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):   
        """
        Initialize the Retriever with embedding and ranking models.
        
        Args:
            embedder_model_name (str): Name of the sentence transformer model for embeddings.
                                      Defaults to 'all-MiniLM-L6-v2'.
            ranker_model_name (str): Name of the cross-encoder model for re-ranking.
                                    Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        """
        self.embedder = Embedder(embedder_model_name)
        self.document_ranker = CrossEncoder(ranker_model_name, activation_fn=torch.nn.Sigmoid())
        self.vector_store = VectorStore()
        self.last_documents = []
        logger.info(f"Retriever initialized with embedder: {embedder_model_name} and ranker: {ranker_model_name}")

    def retrieve(self, query: str, n_results: int = 10, threshold: float = 0.5) -> list[Document]:
        """
        Retrieve and re-rank documents based on the query.
        
        This method performs a two-stage retrieval: first retrieving candidate
        documents using semantic search, then re-ranking them using a cross-encoder.
        
        Args:
            query (str): The search query.
            n_results (int): Number of documents to retrieve. Defaults to 10.
            threshold (float): Minimum cross-encoder score for accepting documents.
                              Defaults to -3.0. Lower values are more permissive.
        Returns:
            list[Document]: List of retrieved documents, sorted by relevance score.
        """
        documents = self.vector_store.query(query, n_results)
        logger.debug(f"Retrieved {len(documents)} documents")
        # If no documents are retrieved, return the default document
        if len(documents) == 0:
            logger.warning(f"Returning default document because "
                           f"no documents retrieved for query: {query}")
            return [DEFAULT_DOCUMENT]
        de_duped_documents = self.de_duplicate_documents(documents)
        reordered_documents = self.reorder_documents(de_duped_documents, query)
        # If the top document is not relevant, return the default doc
        if len(reordered_documents) == 0:
            logger.warning(f"Returning default document because "
                           f"no documents in list after de-duplication: {query}")
            return [DEFAULT_DOCUMENT]
        top_score = reordered_documents[0].rank
        # Implementing delta score - if the top score is much higher than the second score
        # return it as the correct document even if the score is not high enough
        if len(reordered_documents) > 1:
            second_score = reordered_documents[1].rank
        else:
            second_score = 0.0
        logger.warning(f"Top score: {top_score}, second score: {second_score}, delta: {top_score-second_score}")
        if top_score < threshold and top_score-second_score < 0.1:
            logger.warning(f"Returning default document due to low rank after reordering "
                           f"score:{reordered_documents[0].rank} < {threshold}: {query}")
            return [INSUFFICIENT_RELEVANCE_DOCUMENT]
        self.last_documents = reordered_documents
        return self.last_documents

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
    
    def de_duplicate_documents(self, documents: list[Document], threshold:float = 0.95) -> list[Document]:
        """
        Remove duplicate documents from the list.  Documents are duplicates if they have
        data that is syntactically identical (approximate similarity is .95 by default)
        """
        texts = [doc.data for doc in documents]
        logger.debug(f"De-duplicating {len(texts)} documents")
        embeddings = self.embedder.embed_batch(texts)
        np_embeddings = np.array(embeddings)
        keep_indexes = []
        keep_embeddings = []

        for i, embed in enumerate(np_embeddings):
            if keep_embeddings:
                similarities = cosine_similarity([embed], keep_embeddings)[0]
                if np.any(similarities > threshold):
                    continue
            keep_indexes.append(i)
            keep_embeddings.append(embed)

        logger.debug(f"Kept {len(keep_indexes)} documents after de-duplication")
        return [documents[i] for i in keep_indexes]
    
    def clear_last_documents(self):
        """
        Clear the last retrieved documents from memory.
        
        This method resets the last_documents list to an empty state. It's typically
        used for test isolation or when starting a new retrieval session to ensure
        no stale document references remain from previous queries.
        """
        self.last_documents = []

    