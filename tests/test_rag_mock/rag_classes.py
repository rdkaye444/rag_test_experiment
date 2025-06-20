from typing import List, Dict, Any

class Retriever:
    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """Retrieve documents based on query"""
        raise NotImplementedError("Retriever is not implemented")

class Generator:
    """Generate response based on documents"""
    def generate(self, query:str, documents: List[Dict[str, str]]) -> str:
        raise NotImplementedError("Generator not implemented")

class RAG:
    def __init__(self, retriever: Retriever, generator: Generator)-> None:
        self.retriever = retriever
        self.processor = generator

    def run(self, query: str)-> Dict[str, Any]:
        documents = self.retriever.retrieve(query)
        answer = self.processor.generate(query, documents)
        return {'documents': documents, 'answer': answer}
    
