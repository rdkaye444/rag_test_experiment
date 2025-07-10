from rag.judge import Judge, JudgeResult
import pytest
from schema.generator_config import GeneratorConfig


def test_generate_platypus_strict_should_judge_true(pipeline_factory):
    gen_config = GeneratorConfig(mode="strict")
    pipeline, retriever, _  = pipeline_factory(gen_config)
    response = pipeline.run("Why is a platypus so weird?")
    judge = Judge()
    result = judge.judge(response, retriever.last_documents)
    if result != JudgeResult.TRUE:
        explanation = judge.explain(response, retriever.last_documents)
        pytest.fail(f"Judgment was {result.name}. Explanation:\n{explanation}")
