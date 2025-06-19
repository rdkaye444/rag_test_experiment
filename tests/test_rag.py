import pytest
from tests.rag_classes import Retriever, Generator, RAG
from typing import List, Dict

import pprint

class MockRetriever(Retriever):
    def retrieve(self, query: str)-> List[Dict[str, str]]:
        return [
            {'content': 'Paris is the capital of France'}
        ] if "capital" in query.lower() else []
    
class MockGenerator(Generator):
    def generate(self, query: str, documents: List[Dict[str, str]])-> str:
        return "Paris is the capital of France" if "capital" in query.lower() and "Paris" in documents[0]["content"] else "I don't know"

def test_retriever_returns_expected_documenets():
    retriever = MockRetriever()
    docs = retriever.retrieve("What is the capital of France?")
    assert any("Paris" in doc["content"] for doc in docs)

def test_retriever_returns_no_docs():
    retriever = MockRetriever()
    docs = retriever.retrieve("What is the color of the sky?")
    assert len(docs) == 0

def test_generate_returns_context_and_answer():
    generator = MockGenerator()
    docs = [{'content': 'Paris is the capital of France'}]
    query = "What is the capital of France?"
    result = generator.generate(query, docs)
    assert result == "Paris is the capital of France"

def test_generate_returns_no_answer():
    generator = MockGenerator()
    docs = [{'content': 'Paris is the capital of France'}]
    query = "What is the most important city of Germany?"
    result = generator.generate(query, docs)
    assert result == "I don't know"

def test_retrieve_and_generate_returns_context_and_answer():
    retriever = MockRetriever()
    generator = MockGenerator()
    rag = RAG(retriever, generator)
    query = "What is the capital of France?"
    result = rag.run(query)
    pprint.pprint(result)
    assert result == {'answer': 'Paris is the capital of France',
 'documents': [{'content': 'Paris is the capital of France'}]}