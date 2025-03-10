from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

sentiment_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
        You are a sentiment analysis expert. Analyze the following sentence and return:
        1. The overall sentiment: Positive, Negative, Neutral, or Mixed.
        2. An explanation describing why it was classified that way.

        Sentence: "{text}"

        Response Format (JSON):
        {{
            "sentiment_label": "...",
            "explanation": "..."
        }}
    """
)

def sentiment_chain(llm):
    return LLMChain(llm=llm, prompt=sentiment_prompt, output_key="sentiment_result")