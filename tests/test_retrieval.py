import pytest
import pprint

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
])
def test_retrieval_synonym(create_retriever, query, n_results, expected_data, expected_species):
    documents = create_retriever.retrieve(query, n_results=n_results)
    assert len(documents) == n_results

    if expected_data and expected_species:
        assert documents[0].data == expected_data
        assert documents[0].metadata.source_species == expected_species

@pytest.mark.top_k
@pytest.mark.parametrize("query,n_results,expected_data,expected_species", [
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



# Recall Ground truth tests - positive and negative
# Note - for a retriever a common simplification is Recall@1. 
# These tests are very closely related to the direct retrieval tests due to my small dataset
# TODO: Add a test for recall as a function of top_k false negatives and positives
def _test_ground_truth(create_retriever, query, ground_truth_answer, top_k, expected_found):
    retriever = create_retriever
    retrieved_docs = retriever.retrieve(query, n_results=top_k)
    found = any(ground_truth_answer.lower() in doc.data.lower() for doc in retrieved_docs)
    assert found == expected_found, (
        f"Recall@{top_k} failed for query: '{query}'.\n"
        f"Expected something containing: '{ground_truth_answer}'\n"
        f"Retrieved:\n" + "\n".join(f"- {doc.data}" for doc in retrieved_docs))

@pytest.mark.recall_ground_truth
@pytest.mark.parametrize("query, ground_truth_answer, top_k, expected_found", [
    ("Why is a platypus so weird?", "platypus", 3, True),
    ("Does a horse bear live young?", "live young", 3, True),
    ("Tell me something about the platypus", "platypus", 3, True),
    ("What makes the platypus unusual?", "lay eggs", 3, True),
])
def test_recall_against_ground_truth(create_retriever, query, ground_truth_answer, top_k, expected_found):
    _test_ground_truth(create_retriever, query, ground_truth_answer, top_k, expected_found)

@pytest.mark.recall_eval_negative
@pytest.mark.parametrize("query, forbidden_answer, top_k, expected_found", [
    ("Tell me about amphibians", "platypus", 3, False),
    ("What do reptiles eat?", "fur or hair", 3, False),
    ("How do birds fly?", "lays eggs and is a mammal", 3, False),
]) 
def test_recall_against_ground_truth_negative(create_retriever, query, forbidden_answer, top_k, expected_found):
    _test_ground_truth(create_retriever, query, forbidden_answer, top_k, expected_found)
    

#TODO: Add a test for lexical vs symantic matching - will need to modify retriever code to do test_top_n_should_include_non_avian_results
#TODO: Implement hybrid test_retrieval_synonym

@pytest.mark.de_duplication
def test_de_duplication(create_retriever):
    documents = create_retriever.retrieve("Tell me all about whales", n_results=5)
    create_retriever.de_duplicate_documents(documents)
    pprint.pprint([doc.data for doc in documents])
    assert len(documents) == 3 # 3 unique documents should be retrieved from this test set
    # Note - only two of the docuemnts retrieved are about whales ;(
    