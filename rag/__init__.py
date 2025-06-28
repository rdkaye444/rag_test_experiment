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
