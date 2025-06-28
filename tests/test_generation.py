from rag.retriever import Retriever
import pytest
import pprint

@pytest.fixture(scope="session")
def create_retriever():
    retriever = Retriever()
    retriever.vector_store.seed_documents()
    return retriever

def test_retrieve_platypus(create_retriever):
    documents = create_retriever.retrieve("Why is a platypus so weird?")
    pprint.pprint(documents)
    assert documents[0].metadata.source_species == "mammal"
    assert documents[0].data == "Platypus are mammals that lay eggs.  They are very strange mammals."



