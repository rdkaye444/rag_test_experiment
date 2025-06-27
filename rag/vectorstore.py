from schema.document import Document
from rag.embedding import Embedder
from sentence_transformers import CrossEncoder
from pathlib import Path
import json
import chromadb 
from datetime import datetime

SEED_DATA_PATH = Path('data/seed_data.json')
COLLECTION_NAME = 'seed_data'

class VectorStore:
    def __init__(self, embedder_model_name:str = 'all-MiniLM-L6-v2'):        
        self.embedder = Embedder(embedder_model_name)
        self.client=chromadb.EphemeralClient()
        self.collection = self.client.create_collection(name=COLLECTION_NAME,
                                                        embedding_function=self.embedder.embed,
                                                        metadata={'source': 'test',
                                                                  'created_at': datetime.now().isoformat()})

    def seed_documents(self):
        if not SEED_DATA_PATH.exists():
            raise FileNotFoundError(f"Seed data file not found at {SEED_DATA_PATH}")
        with open(SEED_DATA_PATH, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.add_document(Document(**data))

    def query(self, query: str, n_results: int = 10) -> list[Document]:
        results =  self.collection.query(query_texts=[query], n_results=n_results)
        doc_results = zip(results['ids'][0],results['documents'][0],results['metadatas'][0])
        return [Document(id=id, data=data, metadata=metadata) for id, data, metadata in doc_results]
    
    def add_documents(self, documents: list[Document]):
        self.collection.add(
            documents=[doc.data for doc in documents],
            ids=[doc.id for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )       
