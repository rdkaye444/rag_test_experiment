"""
RAG (Retrieval-Augmented Generation) package.

This package provides a complete implementation of a RAG system with the following components:

- Embedding: Text embedding generation using sentence transformers
- Retriever: Document retrieval with semantic search and re-ranking
- Generator: Response generation based on retrieved documents
- VectorStore: Vector database for storing and querying document embeddings
- Pipeline: End-to-end RAG pipeline that combines retrieval and generation

The system is designed to be modular and extensible, allowing for easy customization
of individual components while maintaining a clean interface for the complete pipeline.
"""
import logging

from rag.config import THIRD_PARTY_LOGGING_LEVEL
from rag.logger import setup_logger

# silence noisy loggers by setting them to WARNING
logging.getLogger().setLevel(THIRD_PARTY_LOGGING_LEVEL)

# Setup the application logger
setup_logger(__name__)