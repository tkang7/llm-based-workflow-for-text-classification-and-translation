### sentiment analysis
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI

# Construct the Tools agent
# agent = create_tool_calling_agent(llm = model, tools = [sentiment_tool], prompt = prompt)

class SentimentAnalysis:
    def __init__(self, llm=None):
        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    def analyze(self, text, use_openAI=True):
        prompt = (
            f"What is the sentiment (Positive, Negative, Neutral) of the following sentence? "
            f"Explain your reasoning in detail.\n\n"
            f"Sentence: \"{text}\""
        )

        if use_openAI:
            response = self.llm.predict(prompt)
        else:
            # self.llm is a transformers pipeline
            outputs = self.llm(prompt, max_length=512, do_sample=True)
            
            response = outputs[0]['generated_text']
            
            # If the prompt is repeated in the response, trim it:
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
        return response