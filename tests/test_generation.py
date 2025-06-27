from rag.retriever import Retriever
import pytest

@pytest.fixture(scope="session")
def create_retriever():
    retriever = Retriever()
    retriever.vector_store.seed_documents()
    return retriever

def test_retrieve(create_retriever):
    documents = create_retriever.retrieve("What is the wierdest animal?")
    assert documents[0].metadata.tags == ["mammals", "aquatic"]
    assert documents[0].data == "Platypus are mammals that lay eggs.  They are very strange mammals."





