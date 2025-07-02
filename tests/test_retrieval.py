import pytest

@pytest.mark.parametrize("query,n_results,expected_data,expected_species", [
    # Direct retrievals
    ("Why is a platypus so weird?", 1, "Platypus are mammals that lay eggs.  They are very strange mammals.", "mammal"),
    ("Does a horse bear live young?", 1, "A horse is a mammal.  Mammals are warm-blooded animals that have fur or hair.  They give birth to live young.", "mammal"),
    
    # Synonym/semantic matches
    ("Does a mare bear live young?", 1, "A horse is a mammal.  Mammals are warm-blooded animals that have fur or hair.  They give birth to live young.", "mammal"),
    ("Does an equine bear live young?", 1, "A horse is a mammal.  Mammals are warm-blooded animals that have fur or hair.  They give birth to live young.", "mammal"),
    ("Does a bird lay eggs?", 1, "Birds lay eggs to reproduce.  Eggs are delicious", "avian"),
    ("Which sea creature sings?", 1, "Humpback whales are known for their complex songs. These marine mammals use sound to communicate.", "mammal"),
    
    # Top-k controls
    ("How many bird species do you know about?", 3, None, None),  # We’ll just check len==3
])
def test_retriever_parametrized(create_retriever, query, n_results, expected_data, expected_species):
    documents = create_retriever.retrieve(query, n_results=n_results)
    assert len(documents) == n_results

    if expected_data and expected_species:
        assert documents[0].data == expected_data
        assert documents[0].metadata.source_species == expected_species

def test_top_n_should_include_non_avian_results(create_retriever):
    documents = create_retriever.retrieve("How many bird species do you know about?", n_results=10)
    assert len(documents) == 10
    assert not all(doc.metadata.source_species == "avian" for doc in documents)

def test_top_n_all_results_should_be_birds(create_retriever):
    documents = create_retriever.retrieve("How many bird species do you know about?", n_results=5)
    assert all(doc.metadata.source_species == "avian" for doc in documents)

@pytest.mark.skip(reason="Fallback logic not yet implemented")
def test_fallback_low_recall_domain(create_retriever):
    documents = create_retriever.retrieve("What is the meaning of life?")
    assert documents[0].data == "Platypus are mammals that lay eggs.  They are very strange mammals."


def test_should_fail_with_exception_verify_logging(create_retriever):
    raise Exception("This is a test exception")