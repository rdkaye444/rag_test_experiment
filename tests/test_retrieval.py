import pytest
import pprint

@pytest.mark.direct_retrieval
@pytest.mark.parametrize("query,n_results,expected_data,expected_species", [
    # Basic retrievals
    ("Do platypuses lay eggs?", 1, "Platypus are mammals that lay eggs.  They are very strange mammals.", "mammal"),
    ("Are penguins flightless?",1, "Penguins are flightless birds that lay eggs and raise their chicks on icy terrain.", "avian"),
    ("Does a horse have live young?", 1, "A horse is a mammal.  Mammals are warm-blooded animals that have fur or hair.  They give birth to live young.", "mammal"),
    # Abstract references to specific species retrieval
    ("Tell me something about the platypus", 1, "Platypus are mammals that lay eggs.  They are very strange mammals.", "mammal"),
    ("Tell me about crocodiles", 1, "Crocodiles are reptiles that lay eggs in nests near water. They are cold-blooded and have scaly skin.", "reptile"),
    ("What is special about bats?", 1, "Bats are the only mammals capable of sustained flight. They give birth to live young.", "mammal"),
    ("Is there anything I should know about the platypus?", 1, "Platypus are mammals that lay eggs.  They are very strange mammals.", "mammal"),
])
def test_direct_retrieval_should_return_expected_data(create_retriever, query, n_results, expected_data, expected_species):
    documents = create_retriever.retrieve(query, n_results=n_results)
    assert len(documents) == n_results

    if expected_data and expected_species:
        assert documents[0].data == expected_data
        assert documents[0].metadata.source_species == expected_species


@pytest.mark.synonym
@pytest.mark.parametrize("query,n_results,expected_data,expected_species,threshold", [
    ("Does a mare give birth to live young?", 2, "A horse is a mammal.  Mammals are warm-blooded animals that have fur or hair.  They give birth to live young.", "mammal", .4),
    ("Do horses give birth to live young?", 2, "A horse is a mammal.  Mammals are warm-blooded animals that have fur or hair.  They give birth to live young.", "mammal", .4),
    ("Does an equine give birth to live young?", 1, "A horse is a mammal.  Mammals are warm-blooded animals that have fur or hair.  They give birth to live young.", "mammal", .4),
    ("Do avians lay eggs?", 1, "Birds lay eggs to reproduce.  Eggs are delicious", "avian", .4),
    ("Does a duck-billed platypus lay eggs?", 1, "Platypus are mammals that lay eggs.  They are very strange mammals.", "mammal", .4),
    ("Do macropods carry their young in pouches?", 1, "Kangaroos are marsupials. They carry their young in pouches and give birth to live offspring.", "mammal", .4),
    ("Do cetaceans nurse their calves?", 1, "Whales are marine mammals. They give birth to live calves and nurse them underwater.", "mammal", .4),
    ("Do crocodilian lay eggs?", 1, "Crocodiles are reptiles that lay eggs in nests near water. They are cold-blooded and have scaly skin.", "reptile", .4),
    ("Does a lizard spawn?", 1, "Lizards are reptiles that can regrow their tails and usually lay soft-shelled eggs.", "reptile", .4),
    ("Which reptiles are oviparous?", 1, "Crocodiles are reptiles that lay eggs in nests near water. They are cold-blooded and have scaly skin.", "reptile", .01),
])
def test_retrieval_synonym(create_retriever, query, n_results, expected_data, expected_species, threshold):
    documents = create_retriever.retrieve(query, n_results=n_results, threshold=threshold)
    assert len(documents) == n_results

    if expected_data and expected_species:
        assert documents[0].data == expected_data
        assert documents[0].metadata.source_species == expected_species

@pytest.mark.top_k
@pytest.mark.parametrize("query,n_query,de_dupe_doc_count,expected_count,expected_species,threshold", [
("How many bird species do you know about?", 13, 13 ,7, ["avian"],0.0),
("How many mammal species do you know about?", 15 , 13, 8, ["mammal"],0.0),
("How many bird and mammal species do you know about?", 11, 9, 9, ["avian", "mammal"],0.0),
])
def test_retrieval_top_k_low_threshold_should_return_correct_set_of_results(create_retriever, query, n_query,de_dupe_doc_count, expected_count, expected_species, threshold):
    documents = create_retriever.retrieve(query, n_results=n_query, threshold=threshold)
    assert de_dupe_doc_count == len(documents), f"Expected {de_dupe_doc_count} de-dupedresults, got {len(documents)}"
    filtered_results = [doc for doc in documents if doc.metadata.source_species in expected_species]
    assert len(filtered_results) >= expected_count, f"Expected {expected_count} {expected_species} results, got {len(filtered_results)}"

@pytest.mark.top_k
def test_top_n_should_include_non_avian_results(create_retriever):
    """When reducing threshold to -1 we should get all results"""
    documents = create_retriever.retrieve("How many bird species do you know about?", n_results=10, threshold=-1)
    assert len(documents) == 10
    assert not all(doc.metadata.source_species == "avian" for doc in documents)

@pytest.mark.top_k
def test_top_n_all_results_should_be_birds(create_retriever):
    """When reducing threshold to -1 we should get all 5 results in test data set of type avian"""
    documents = create_retriever.retrieve("How many bird species do you know about?", n_results=5, threshold=-1)
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
    # Note - because my test set is so small, I have to restrict the number of results
    # to get a passing test.clear

    documents = create_retriever.retrieve(query, n_results=2)
    for document in documents:
        assert negated_search_term not in document.data.lower()


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
    

@pytest.mark.ambiguous_retrieval
@pytest.mark.parametrize("query,n_results,threshold", [
    ("What animals do you like?", 5, -1),
    ("Tell me about life forms",6, -1)
])
def test_ambiguous_query_retrieval_no_threshold_returns_n_results(create_retriever,query, n_results, threshold):
    """When reducing the threshold to -1 we should get n results"""
    documents = create_retriever.retrieve(query, n_results, threshold)
    assert len(documents) == n_results, f"Expected {n_results} due to threshold of {threhold} results, got {len(documents)}"

@pytest.mark.ambiguous_retrieval
@pytest.mark.parametrize("query,n_results,threshold", [
    ("What animals do you like?", 5, .5),
    ("Tell me about life forms",6, .5)
])
def test_ambiguous_retrieval_with_threshold_returns_default_document(create_retriever,query, n_results, threshold):
    """When using a realistic threshold we should get 1 result - the default document"""
    documents = create_retriever.retrieve(query, n_results, threshold)
    pprint.pprint([doc.data for doc in documents])
    assert len(documents) == 1 

@pytest.mark.low_recall_domain
@pytest.mark.parametrize("query,n_results,threshold", [
    ("When do ducklings learn to swim?", 1, .9),
    ("Where do penguins have kids?", 1, .9),
    ("Which animals carry their young in pouches?", 1, .9),
])
def test_low_recall_domain(create_retriever, query, n_results, threshold):
    """This test is for very high detail retrieval - we should get a single document
    that is in the test data set containing this detail with a very high threshold"""
    documents = create_retriever.retrieve(query, n_results, threshold)
    assert len(documents) == 1 

@pytest.mark.fallback
@pytest.mark.parametrize("query,n_results,threshold", [
    ("What would happen if microwave blueberry zero nonsenese?", 1, 0.1),
    ("asdfew?", 1, 0.1),
    ("Tell me about everything", 1, 0.1),
])
def test_low_recall_domain(create_retriever, query, n_results, threshold):
    """This test verifies that fallback logic is working for garbage queries
    even when the threshold is low"""
    documents = create_retriever.retrieve(query, n_results, threshold)
    assert len(documents) == 1 
    assert documents[0].id == 'insufficient_relevance'

