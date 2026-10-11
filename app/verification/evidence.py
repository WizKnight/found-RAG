from app.models.generation import Citation


class EvidenceResolver:
    def __init__(
        self,
        citations: list[Citation],
    ) -> None:
        self._citations = {
            citation.citation_id: citation
            for citation in citations
        }

    def resolve(
        self,
        citation_ids: list[str],
    ) -> list[Citation]:
        return [
            self._citations[citation_id]
            for citation_id in citation_ids
            if citation_id in self._citations
        ]