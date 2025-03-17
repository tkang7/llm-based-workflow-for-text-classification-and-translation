### sentiment analysis
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Construct the Tools agent
# agent = create_tool_calling_agent(llm = model, tools = [sentiment_tool], prompt = prompt)

class SentimentAnalysisSubAgent:
    def __init__(self):
        
        self.prompt_description = f"""
        You are a sentiment analysis expert. Analyze the following sentence and return:
        1. The overall sentiment: Positive, Negative, Neutral, or Mixed.
        2. An explanation describing why it was classified that way.

        """
        self.prompt_output_format = f"""
        Response Format (JSON):
        {{
            "sentiment_label": "...",
            "explanation": "..."
        }}
        """

    def analyze(self, question, llm=None):
        full_prompt = self.prompt_description + f"\nSentence: {question}\n" + self.prompt_output_format

        print("Starting Sentiment Analysis...")
        if llm:
            result = llm(full_prompt, max_length=512, batch_size=4, do_sample=False)
        else:
            print("No model provided. Using Default Sentiment Analysis Model")
            sentiment_pipeline = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")
            result = sentiment_pipeline(full_prompt)
        print("Finished Sentiment Analysis")

        return result