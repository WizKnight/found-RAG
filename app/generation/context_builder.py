from app.models.generation import Citation
from app.models.retrieval import RetrievalResult


class ContextBuilder:
    def build(
        self,
        results: list[RetrievalResult],
    ) -> tuple[str, list[Citation]]:
        context_parts: list[str] = []
        citations: list[Citation] = []

        for index, result in enumerate(results, start=1):
            citation_id = f"CIT-{index}"

            citation = Citation(
                citation_id=citation_id,
                chunk_id=str(result.chunk_id),
                document_id=(
                    str(result.document_id)
                    if result.document_id
                    else None
                ),
                page_number=result.page_number,
                section=result.section,
                text=result.text,
            )

            citations.append(citation)

            context_parts.append(
                f"[{citation_id}]\n"
                f"{result.text}"
            )

        return "\n\n".join(context_parts), citations