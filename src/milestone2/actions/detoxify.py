from langchain_ollama import ChatOllama
class Detoxifier():
    def __init__(self):
        self.detox_llm = ChatOllama(
            model="detoxllm_ubc_query_only",
            temperature=0
        )

        print("Loaded Ollama UBC Detox Model from local Ollama")

    def detoxify(self, text):
        prompt = (
            "### Instruction:\n"
            "Detoxify the following sentence while keeping its meaning intact.\n\n"
            f"### Input:\n{text}\n\n### Response:"
        )
        response = self.detox_llm.invoke(prompt)
        return response.content.strip()