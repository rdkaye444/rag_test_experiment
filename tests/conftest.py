import pytest

from rag.generator import Generator
from rag.pipeline import RagPipeline
from rag.retriever import Retriever


@pytest.fixture(scope="session")
def create_retriever():
    retriever = Retriever()
    retriever.vector_store.seed_documents()
    return retriever

@pytest.fixture(scope="session")
def create_pipeline(create_retriever):
    generator = Generator()
    return RagPipeline(create_retriever, generator)