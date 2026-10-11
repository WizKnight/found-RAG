from functools import lru_cache

from app.core.config import get_settings
from app.generation.llm import LLM
from app.generation.openai_llm import OpenAILLM


@lru_cache
def get_llm() -> LLM:
    settings = get_settings()

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when "
                "LLM_PROVIDER=openai"
            )

        if not settings.llm_model:
            raise ValueError(
                "LLM_MODEL is required when "
                "LLM_PROVIDER=openai"
            )

        return OpenAILLM(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
        )

    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider}"
    )