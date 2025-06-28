"""
VectorStore module for RAG (Retrieval-Augmented Generation) system.

This module provides functionality for storing and querying document embeddings
using ChromaDB. It handles document ingestion, vector storage, and semantic search
operations.
"""

from schema.document import Document, MetaData
from rag.embedding import Embedder, ChromaEmbedder
from pathlib import Path
import json
import chromadb 
from datetime import datetime
import pprint

SEED_DATA_PATH = Path('data/seed_data.jsonl')
COLLECTION_NAME = 'seed_data'

class VectorStore:
    """
    Vector database for storing and querying document embeddings.
    
    This class provides an interface to ChromaDB for storing document embeddings
    and performing semantic search queries. It handles document ingestion from
    JSONL files and provides methods for adding documents and querying the database.
    
    Attributes:
        embedder (Embedder): The embedding model for generating document vectors.
        client (chromadb.EphemeralClient): The ChromaDB client instance.
        collection (chromadb.Collection): The document collection in ChromaDB.
    """
    
    def __init__(self, embedder_model_name: str = 'all-MiniLM-L6-v2'):        
        """
        Initialize the VectorStore with an embedding model and ChromaDB collection.
        
        Args:
            embedder_model_name (str): Name of the sentence transformer model for embeddings.
                                      Defaults to 'all-MiniLM-L6-v2'.
        """
        self.embedder = Embedder(embedder_model_name)
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(name=COLLECTION_NAME,
                                                        embedding_function=ChromaEmbedder(self.embedder),
                                                        metadata={'source': 'test',
                                                                  'created_at': datetime.now().isoformat()})

    def seed_documents(self):
        """
        Load and add documents from the seed data file.
        
        This method reads documents from the configured seed data JSONL file
        and adds them to the vector store for indexing.

        Note:  The seed data path is hardcoded in this module, but in a real production
        application, this would be a configuration parameter.
        
        Raises:
            FileNotFoundError: If the seed data file doesn't exist.
        """
        if not SEED_DATA_PATH.exists():
            raise FileNotFoundError(f"Seed data file not found at {SEED_DATA_PATH}")
        documents = []
        with open(SEED_DATA_PATH, 'r') as f:
            for line in f:
                data = json.loads(line)
                documents.append(Document(**data))
        self.add_documents(documents=documents)

    def query(self, query: str, n_results: int = 10) -> list[Document]:
        """
        Perform a semantic search query on the vector store.
        
        Args:
            query (str): The search query.
            n_results (int): Number of results to return. Defaults to 10.
        
        Returns:
            list[Document]: List of retrieved documents with their metadata.
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)
        doc_results = zip(results['ids'][0], results['documents'][0], results['metadatas'][0])
        return [Document(id=id, data=data, metadata=metadata) for id, data, metadata in doc_results]
    
    def add_documents(self, documents: list[Document]):
        """
        Add documents to the vector store for indexing.
        
        Args:
            documents (list[Document]): List of documents to add to the vector store.
        """
        self.collection.add(
            documents=[doc.data for doc in documents],
            ids=[doc.id for doc in documents],
            metadatas=[doc.metadata.model_dump() for doc in documents]
        )       
