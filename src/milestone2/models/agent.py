from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from src.milestone2.models.llm import LLM
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import initialize_agent, AgentType

# Tools for the agent
from src.milestone2.analysis.sentiment_analysis import SentimentAnalysis
from src.milestone2.analysis.toxicity_analysis import ToxicityAnalysis
from src.milestone2.actions.translate import Translator
from src.milestone2.actions.detoxify import Detoxifier

class Agent:
    def __init__(self, llm_name="gpt-4"):
        self.llm = LLM().create(key=llm_name)
        
        tools = [
            Tool.from_function(
                func=self.translate_to_english,
                name="TranslateToEnglish",
                description="Use this tool to translate sentences into English."
            ),
            Tool.from_function(
                func=self.sentiment_analysis_with_reason,
                name="SentimentAnalysisWithReason",
                description="Use this tool to analyze sentiment of a sentence. Returns sentiment label and reasoning."
            ),
            Tool.from_function(
                func=self.toxicity_analysis_with_reason,
                name="ToxicityAnalysisWithReason",
                description="Use this tool to analyze toxicity of a sentence. Returns toxicity label and reasoning."
            ),
            Tool.from_function(
                func=self.detoxify_sentence,
                name="DetoxifySentence",
                description="Use this tool to detoxify a toxic sentence and rewrite it without harmful language."
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            # llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def sentiment_analysis_with_reason(self, text: str) -> str:
        return SentimentAnalysis().analyze(text)

    def toxicity_analysis_with_reason(self, text: str) -> str:
        return ToxicityAnalysis().analyze(text)

    def translate_to_english(self, text: str) -> str:
        return Translator().translate(text)
    
    def detoxify_sentence(self, text: str) -> str:
        return Detoxifier().detoxify(text)

    def run(self, text):
        query = (
            f"Here's a sentence: {text}\n"
            "Please translate it to English if necessary, analyze its sentiment with reasons, "
            "analyze its toxicity with reasons, and detoxify it.\n"
            "Return the result strictly as a JSON array in this format:\n\n"
            "[\n"
            "    {\n"
            "        \"sentiment_label\": \"<Sentiment Label>\",\n"
            "        \"explanation\": \"<Explain why the sentiment was classified as such>\"\n"
            "    },\n"
            "    {\n"
            "        \"toxicity_label\": \"<Toxic/Non-Toxic>\",\n"
            "        \"explanation\": \"<Explain why the sentence was considered toxic or not>\"\n"
            "    },\n"
            "    {\n"
            "        \"detoxified_text\": \"<Detoxified Sentence>\",\n"
            "        \"explanation\": \"<Explain how you modified the sentence to make it non-toxic>\"\n"
            "    }\n"
            "]"
        )

        response = self.agent.run(query)

        return response
    
if __name__ == "__main__":
    test_llm = "gpt-4"
    prod_llm = "mistralai/Mistral-7B-v0.1"
    agent = Agent(llm_name=test_llm) 
    
    
    input_sentence = "You're so stupid and annoying! I can't stand you."

    response = agent.run(input_sentence)

    print("\nAgent Response:\n")
    print(response)