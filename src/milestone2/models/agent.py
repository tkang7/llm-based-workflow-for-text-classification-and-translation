from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from src.milestone2.models.llm import LLM
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import initialize_agent, AgentType

# Tools for the agent
from src.milestone2.analysis.sentiment_analysis import SentimentAnalysisSubAgent

class Agent:
    def __init__(self, llm):
        self.llm = llm
        self.sentiment_agent = SentimentAnalysisSubAgent()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful LLM Agent that selects from a variety of tools to created a structured response. You have access to:"
                    "\n1. Sentiment Analysis Tool"
                    "\n2. Toxicity Analysis Tool"
                    "\n3. Language Translation Tool"
                    "\n4. Sentence Detoxification Tool"

                    "\n\nUse any number of tools for the given input."
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        tools = [
            Tool(
                name="Sentiment Analysis Tool",
                func=self.sentiment_analysis,
                description="Analyzes the sentiment of a given text and returns the sentiment label.",
                return_direct=True
            ),
        ]

        # print(tools[0].func("I am happy"))

        self.agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.SELF_ASK_WITH_SEARCH,
            verbose=True,
            agent_kwargs={
                "prompt": prompt 
            },
        )
    
    def sentiment_analysis(self, query):
        # print("sentiment --->")
        # llm = LLM()
        # sentiment_llm = llm.create("medium")
        return self.sentiment_agent.analyze(query, llm=self.llm)
    
    def run(self, query):
        return self.agent_executor.invoke(query)

# class Agent:
#     def __init__(self, llm):
#         self.llm = llm
#         self.sentiment_agent = SentimentAnalysisSubAgent()
        
#         # Define tools
#         self.tools = [
#             Tool(
#                 name="Sentiment Analysis",
#                 func=self.sentiment_analysis,
#                 description="Useful for analyzing the sentiment of text"
#             ),
#         ]
        
#         # Initialize agent
#         print("Initializing agent...")
#         self.agent = initialize_agent(
#             self.tools,
#             self.llm,
#             verbose=True
#         )
#         print("Agent initialized...")
    
#     def sentiment_analysis(self, query):
#         return self.sentiment_agent.analyze(self.llm, query)
    
#     def run(self, query):
#         return self.agent.run(query)