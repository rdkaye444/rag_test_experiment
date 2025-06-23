from pytest import fixture
from mock.retriever import Retriever
from unittest.mock import MagicMock
from schema.document import Document

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
