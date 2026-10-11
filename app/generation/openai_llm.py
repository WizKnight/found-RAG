from openai import OpenAI

from app.generation.llm import LLM


class OpenAILLM(LLM):
    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
        )

        return response.output_text