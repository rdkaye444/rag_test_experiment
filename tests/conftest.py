import pytest
from rag.retriever import Retriever

@pytest.fixture(scope="session")
def create_retriever():
    retriever = Retriever()
    retriever.vector_store.seed_documents()
    return retriever