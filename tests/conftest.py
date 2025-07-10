import pytest

from rag.generator import Generator
from rag.pipeline import RagPipeline
from rag.retriever import Retriever
from schema.generator_config import GeneratorConfig


@pytest.fixture(scope="session")
def create_retriever():
    retriever = Retriever()
    retriever.vector_store.seed_documents()
    return retriever

@pytest.fixture(scope="function")
def pipeline_factory(create_retriever):
    def _factory(config: GeneratorConfig | None = None):
        create_retriever.clear_last_documents()
        if config is None:
            config = GeneratorConfig(mode="loose")
        generator = Generator(config)
        return RagPipeline(create_retriever, generator), create_retriever, generator
    return _factory
