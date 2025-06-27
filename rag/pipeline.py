from rag.retriever import Retriever
from rag.generator import Generator


class RagPipeline:
    def __init__(self, query:str, retriever:Retriever, generator:Generator):
        self.query = query
        self.generator = generator
        self.retriever = retriever

    def run(self):
        documents = self.retriever.retrieve(self.query)
        return self.generator.generate(self.query, documents)





