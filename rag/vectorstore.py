from schema.document import Document, MetaData
from rag.embedding import Embedder, ChromaEmbedder
from pathlib import Path
import json
import chromadb 
from datetime import datetime
import pprint

SEED_DATA_PATH = Path('data/seed_data.jsonl')
COLLECTION_NAME = 'seed_data'

class VectorStore:
    def __init__(self, embedder_model_name:str = 'all-MiniLM-L6-v2'):        
        self.embedder = Embedder(embedder_model_name)
        self.client=chromadb.EphemeralClient()
        self.collection = self.client.create_collection(name=COLLECTION_NAME,
                                                        embedding_function=ChromaEmbedder(self.embedder),
                                                        metadata={'source': 'test',
                                                                  'created_at': datetime.now().isoformat()})

    def seed_documents(self):
        if not SEED_DATA_PATH.exists():
            raise FileNotFoundError(f"Seed data file not found at {SEED_DATA_PATH}")
        documents = []
        with open(SEED_DATA_PATH, 'r') as f:
            for line in f:
                data = json.loads(line)
                documents.append(Document(**data))
        self.add_documents(documents=documents)

    def query(self, query: str, n_results: int = 10) -> list[Document]:
        results =  self.collection.query(query_texts=[query], n_results=n_results)
        doc_results = zip(results['ids'][0],results['documents'][0],results['metadatas'][0])
        return [Document(id=id, data=data, metadata=metadata) for id, data, metadata in doc_results]
    
    def add_documents(self, documents: list[Document]):
        self.collection.add(
            documents=[doc.data for doc in documents],
            ids=[doc.id for doc in documents],
            metadatas=[doc.metadata.model_dump() for doc in documents]
        )       
