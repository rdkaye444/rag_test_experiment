from rag.judge import Judge, JudgeResult
import pytest

def test_generate_platypus(pipeline_factory):
    pipeline, retriever, _  = pipeline_factory()
    response = pipeline.run("Why is a platypus so weird?")
    judge = Judge()
    result = judge.judge(response, retriever.last_documents)
    if result != JudgeResult.TRUE:
        explanation = judge.explain(response, retriever.last_documents)
        pytest.fail(f"Judgment was {result.name}. Explanation:\n{explanation}")
