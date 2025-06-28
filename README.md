# RAG Testing Project

A comprehensive testing framework for Retrieval-Augmented Generation (RAG) systems, designed to evaluate and validate the performance of document retrieval and response generation components.

## Overview

This project implements a modular RAG system with automated testing capabilities. It provides a complete pipeline for document retrieval, semantic search, re-ranking, and response generation, along with comprehensive test suites to validate system performance.

## Features

- **Modular RAG Architecture**: Clean separation of concerns with dedicated components for embedding, retrieval, generation, and vector storage
- **Semantic Search**: Uses sentence transformers for document embedding and similarity search
- **Re-ranking**: Implements cross-encoder models for improved document relevance scoring
- **Vector Database**: ChromaDB integration for efficient document storage and retrieval
- **Comprehensive Testing**: Automated test suites for retrieval and generation components
- **Mock Generator**: Simple generator implementation for testing and development that can easily be extended to add an LLM
- **Pydantic Schemas**: Type-safe data models for documents and queries

## Project Structure

```
rag_testing/
├── data/
│   └── seed_data.jsonl          # Sample documents for testing
├── rag/                         # Core RAG implementation
│   ├── __init__.py              # Package initialization
│   ├── embedding.py             # Text embedding functionality
│   ├── generator.py             # Response generation (mock implementation)
│   ├── pipeline.py              # End-to-end RAG pipeline
│   ├── retriever.py             # Document retrieval with re-ranking
│   └── vectorstore.py           # ChromaDB vector store interface
├── schema/                      # Data models
│   ├── document.py              # Document and metadata schemas
│   └── query.py                 # Query schema for testing
├── tests/                       # Test suites
│   ├── conftest.py              # Pytest configuration and fixtures
│   ├── test_generation.py       # Generation component tests
│   ├── test_retrieval.py        # Retrieval component tests
│   └── utilities/               # Test utilities
├── main.py                      # Main application entry point
├── pyproject.toml               # Project configuration and dependencies
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.11 or higher
- uv package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag_testing
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

4. **Verify installation**:
   ```bash
   python -c "import rag; print('RAG package imported successfully')"
   ```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_retrieval.py

# Run tests with verbose output
pytest -v
```

### Test Structure

- **`tests/test_retrieval.py`**: Tests for document retrieval functionality
- **`tests/test_generation.py`**: Tests for response generation (currently empty)
- **`tests/conftest.py`**: Shared test fixtures and configuration

### Example Test

```python
def test_retrieve_platypus(create_retriever):
    documents = create_retriever.retrieve("Why is a platypus so weird?")
    
    # Verify the most relevant document
    assert documents[0].metadata.source_species == "mammal"
    assert "platypus" in documents[0].data.lower()
```

## Architecture

### Core Components

1. **Embedder** (`rag/embedding.py`)
   - Generates text embeddings using sentence transformers
   - Supports single and batch embedding operations
   - Provides ChromaDB-compatible wrapper

2. **VectorStore** (`rag/vectorstore.py`)
   - Manages document storage and retrieval using ChromaDB
   - Handles document ingestion from JSONL files
   - Provides semantic search capabilities

3. **Retriever** (`rag/retriever.py`)
   - Implements two-stage retrieval: semantic search + re-ranking
   - Uses cross-encoder models for improved relevance scoring
   - Returns ranked document lists

4. **Generator** (`rag/generator.py`)
   - Mock implementation for response generation
   - Maintains prompt history for debugging
   - Designed for easy replacement with LLM integration

5. **Pipeline** (`rag/pipeline.py`)
   - Orchestrates the complete RAG workflow
   - Provides simple interface for end-to-end queries

### Data Models

- **Document**: Represents a document with metadata and content
- **MetaData**: Contains document metadata (title, source, etc.)
- **Query**: Structured query format for testing

## Configuration

### Model Selection

The system uses configurable models for different components:

- **Embedding Model**: `all-MiniLM-L6-v2` (default)
- **Re-ranking Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (default)

You can customize these when initializing components:

```python
retriever = Retriever(
    embedder_model_name='all-mpnet-base-v2',
    ranker_model_name='cross-encoder/ms-marco-MiniLM-L-12-v2'
)
```

### Data Sources

The system loads seed data from `data/seed_data.jsonl`. In production, this would be configurable.

## Development

### Adding New Tests

1. Create test functions in the appropriate test file
2. Use the `create_retriever` fixture for retrieval tests
3. Follow the existing test patterns

### Extending the Generator

The current generator is a mock implementation. To integrate with real LLMs:

1. Replace the `generate` method in `Generator` class
2. Add LLM-specific configuration
3. Update tests to handle the new response format

### Adding New Document Types

1. Extend the `MetaData` schema in `schema/document.py`
2. Update document loading logic in `VectorStore`
3. Add corresponding tests

## Dependencies

### Core Dependencies

- **chromadb**: Vector database for document storage
- **sentence-transformers**: Text embedding models
- **pydantic**: Data validation and serialization
- **numpy**: Numerical operations

### Development Dependencies

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Sentence Transformers for embedding models
- ChromaDB for vector storage
- Pydantic for data validation 