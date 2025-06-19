import pytest
from tests.test_rag_mock.rag_classes import Retriever
from typing import List, Dict


class MockRetriever(Retriever):
    def retrieve(self, query: str)-> List[Dict[str, str]]:
        return [
            {'content': 'Paris is the capital of France'}
        ] if "capital" in query.lower() else []

def test_retriever_returns_expected_documenets():
    retriever = MockRetriever()
    docs = retriever.retrieve("What is the capital of France?")
    assert any("Paris" in doc["content"] for doc in docs)

def test_retriever_returns_no_docs():
    retriever = MockRetriever()
    docs = retriever.retrieve("What is the color of the sky?")
    assert len(docs) == 0