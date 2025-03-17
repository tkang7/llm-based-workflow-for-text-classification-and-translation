from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

toxicity_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
        You are a toxicity detection expert. Analyze the following sentence and return:
        1. Whether the sentence is Toxic or Non-Toxic.
        2. An explanation describing why it was classified that way.

        Sentence: "{text}"

        Response Format (JSON):
        {{
            "toxicity_label": "...",
            "explanation": "..."
        }}
    """
)

def toxicity_chain(llm):
    return LLMChain(llm=llm, prompt=toxicity_prompt, output_key="toxicity_result")