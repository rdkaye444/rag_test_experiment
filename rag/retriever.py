from schema.document import Document
from rag.embedding import Embedder
from sentence_transformers import CrossEncoder
from rag.vectorstore import VectorStore

class Retriever:
    def __init__(self, 
                 embedder_model_name:str = 'all-MiniLM-L6-v2',
                 ranker_model_name:str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.embedder = Embedder(embedder_model_name)
        self.document_ranker = CrossEncoder(ranker_model_name)
        self.vector_store = VectorStore()

    def retrieve(self, query: str, n_results: int = 10)-> list[Document]:
        documents = self.vector_store.query(query, n_results)
        return self.reorder_documents(documents, query)

    def reorder_documents(self, documents: list[Document], query: str) -> list[Document]:
        """
        """
        corpus = [doc.data for doc in documents]
        ranks = self.document_ranker.rank(query, corpus)
        for rank in ranks:
            documents[rank['corpus_id']].score = rank['score']
        return sorted(documents, key = lambda x: x.score, reverse = True)

    