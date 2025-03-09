from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from . import sentiment as sen
from . import toxicity as tox
import os

# Get the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Create ChatOpenAI instance
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

sequential_chain = SequentialChain(
    chains=[sen.sentiment_chain(llm), tox.toxicity_chain(llm)],
    input_variables=["text"],   # Initial input
    output_variables=["sentiment_result", "toxicity_result"],  # Unique outputs
    verbose=True
)

def analyze_text(text):
    result = sequential_chain({"text": text})

    print("Sentiment Analysis Result:", result["sentiment_result"])
    print("Toxicity Detection Result:", result["toxicity_result"])
    
    return result
    