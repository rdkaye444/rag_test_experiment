

def test_generate_platypus(create_pipeline):
    response = create_pipeline.run("Why is a platypus so weird?")
    assert response == "Platypus are mammals that lay eggs.  They are very strange mammals."
