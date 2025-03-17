from langchain.llms.base import LLM
from typing import Any, Optional, List
from pydantic import Extra

class HuggingFacePipelineLLM(LLM):
    pipeline: Any  # <-- Declare it as a class attribute

    @property
    def _llm_type(self) -> str:
        return "huggingface-pipeline"

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline  # Now it's fine!

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        outputs = self.pipeline(prompt, max_length=512, batch_size=1)
        return outputs[0]['generated_text']