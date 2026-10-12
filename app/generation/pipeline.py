from app.generation.context_builder import ContextBuilder
from app.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.generation.llm import LLM
from app.models.generation import GeneratedAnswer
from app.retrieval.pipeline import RetrievalPipeline


class GenerationPipeline:
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        llm: LLM,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.llm = llm
        self.context_builder = (
            context_builder or ContextBuilder()
        )

    def answer(
        self,
        query: str,
        verification_feedback: str | None = None,
    ) -> GeneratedAnswer:
        results = self.retrieval_pipeline.search(query)

        context, citations = (
            self.context_builder.build(results)
        )

        user_prompt = build_user_prompt(
            query=query,
            context=context,
            verification_feedback=verification_feedback,
        )

        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return GeneratedAnswer(
            answer=answer,
            citations=citations,
        )

    def generate(self, query, verification_feedback):
        ...