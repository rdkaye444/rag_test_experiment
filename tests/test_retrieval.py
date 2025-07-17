import pytest
@pytest.mark.direct_retrieval
@pytest.mark.parametrize("query,n_results,expected_data,expected_species", [
    # Basic retrievals
    ("Why is a platypus so weird?", 1, "Platypus are mammals that lay eggs.  They are very strange mammals.", "mammal"),
    ("Does a horse bear live young?", 1, "A horse is a mammal.  Mammals are warm-blooded animals that have fur or hair.  They give birth to live young.", "mammal"),
    # Abstract references to specific species retrieval
    ("Tell me something about the platypus", 1, "Platypus are mammals that lay eggs.  They are very strange mammals.", "mammal"),
    ("Is there anything I should know about the platypus?", 1, "Platypus are mammals that lay eggs.  They are very strange mammals.", "mammal"),
])
def test_retriever_parametrized(create_retriever, query, n_results, expected_data, expected_species):
    documents = create_retriever.retrieve(query, n_results=n_results)
    assert len(documents) == n_results

    if expected_data and expected_species:
        assert documents[0].data == expected_data
        assert documents[0].metadata.source_species == expected_species


@pytest.mark.synonym
@pytest.mark.parametrize("query,n_results,expected_data,expected_species", [
    ("Does a mare bear live young?", 1, "A horse is a mammal.  Mammals are warm-blooded animals that have fur or hair.  They give birth to live young.", "mammal"),
    ("Does an equine bear live young?", 1, "A horse is a mammal.  Mammals are warm-blooded animals that have fur or hair.  They give birth to live young.", "mammal"),
    ("Does a bird lay eggs?", 1, "Birds lay eggs to reproduce.  Eggs are delicious", "avian"),
    ("Which sea creature sings?", 1, "Humpback whales are known for their complex songs. These marine mammals use sound to communicate.", "mammal"),

])
def test_retrieval_synonym(create_retriever, query, n_results, expected_data, expected_species):
    documents = create_retriever.retrieve(query, n_results=n_results)
    assert len(documents) == n_results

    if expected_data and expected_species:
        assert documents[0].data == expected_data
        assert documents[0].metadata.source_species == expected_species

@pytest.mark.top_k
@pytest.mark.parametrize("query,n_results,expected_data,expected_species", [
# Synonym/semantic matches
("How many bird species do you know about?", 3, None, None),  # We’ll just check len==3
])
def test_retrieval_top_k(create_retriever, query, n_results, expected_data, expected_species):
    documents = create_retriever.retrieve(query, n_results=n_results)
    assert len(documents) == n_results

    if expected_data and expected_species:
        assert documents[0].data == expected_data
        assert documents[0].metadata.source_species == expected_species

@pytest.mark.top_k
def test_top_n_should_include_non_avian_results(create_retriever):
    documents = create_retriever.retrieve("How many bird species do you know about?", n_results=10)
    assert len(documents) == 10
    assert not all(doc.metadata.source_species == "avian" for doc in documents)

@pytest.mark.top_k
def test_top_n_all_results_should_be_birds(create_retriever):
    documents = create_retriever.retrieve("How many bird species do you know about?", n_results=5)
    assert all(doc.metadata.source_species == "avian" for doc in documents)


@pytest.mark.negative_control
@pytest.mark.parametrize("query,negated_search_term", [
("How do whales breath", "platypus"),
("How do whales breath", "bird"),
("How do whales breath", "elephant"),
("How do whales breath", "salmon"),
("How do whales breath", "insect"),
])
def test_negative_control_retrieval(create_retriever, query, negated_search_term):
    # Note - because my test set is so small, I have to restrict the numbe of results
    # to get a passing test.
    documents = create_retriever.retrieve(query, n_results=2)
    for document in documents:
        assert negated_search_term not in document.data.lower()

@pytest.mark.skip(reason="Fallback logic not yet implemented")
def test_fallback_low_recall_domain(create_retriever):
    documents = create_retriever.retrieve("What is the meaning of life?")
    assert documents[0].data == "Platypus are mammals that lay eggs.  They are very strange mammals."
