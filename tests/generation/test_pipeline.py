from uuid import uuid4
from typing import cast

from app.generation.pipeline import GenerationPipeline
from app.generation.llm import LLM
from app.models.retrieval import RetrievalResult
from app.retrieval.pipeline import RetrievalPipeline


class FakeRetriever:
    def search(self, query: str) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=uuid4(),
                text="Employees receive 24 days of annual leave.",
                score=0.95,
                rank=1,
                retrieval_method="cross_encoder",
            )
        ]


class FakeLLM(LLM):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        assert "[CIT-1]" in user_prompt

        return (
            "Employees receive 24 days of annual leave. "
            "[CIT-1]"
        )


def test_generation_pipeline() -> None:
    pipeline = GenerationPipeline(
        retrieval_pipeline=cast(RetrievalPipeline, FakeRetriever()),
        llm=FakeLLM(),
    )

    result = pipeline.answer(
        "How many annual leave days do employees receive?"
    )

    assert "24 days" in result.answer
    assert "[CIT-1]" in result.answer

    assert len(result.citations) == 1
    assert result.citations[0].citation_id == "CIT-1"