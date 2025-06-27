from uuid import uuid4
import pytest
from pytest import fixture
from rag.retriever import Retriever
from rag.generator import Generator
from rag.pipeline import RagPipeline
from schema.document import Document, MetaData
from schema.query import Query
from tests.utilities.file_utilities import load_test_data

class StaticMockRetriever(Retriever):
    def __init__(self, documents:list[Document]):
        self.documents = documents
        
    def retrieve(self, query:str):
        return self.documents
    
@fixture
def retriever_factory():
    def _create_retriever(documents:list[Document]):
        return StaticMockRetriever(documents)
    return _create_retriever


class StaticMockGenerator(Generator):
    def __init__(self, query:str, documents:list[Document], expected_semantic_output:str):
        self.query = query
        self.documents = documents
        self.expected_semantic_output = expected_semantic_output

    def generate(self, query:str, documents:list[Document]):
        return self.expected_semantic_output
    
@fixture
def generator_factory():
    def _create_generator(query:str, documents:list[Document], expected_semantic_output:str):
        return StaticMockGenerator(query, documents, expected_semantic_output)
    return _create_generator


def test_pipeline_returns_one_document(retriever_factory, generator_factory):
    query = "can you show me the document?"
    documents = [Document(id=uuid4(), data="document content", metadata=MetaData(title="test", tags=["test"], source="test"))]
    retriever = retriever_factory(documents)
    generator = generator_factory(query, documents, "document content")
    pipeline = RagPipeline(query, retriever, generator)
    assert pipeline.run() == "document content"

@pytest.mark.parametrize("query", load_test_data("queries.jsonl", Query))
def test_pipeline_returns_one_document_parameterized(query, retriever_factory, generator_factory):
    documents = [Document(id=uuid4(), data="document content", metadata=MetaData(title="test", tags=["test"], source="test"))]
    retriever = retriever_factory(documents)
    generator = generator_factory(query.query, documents, "document content")
    pipeline = RagPipeline(query.query, retriever, generator)
    assert pipeline.run() == "document content"