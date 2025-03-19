from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from src.milestone2.models.llm import LLM
from langchain.agents import initialize_agent, AgentType

# Tools for the agent
from src.milestone2.analysis.sentiment_analysis import SentimentAnalysis
from src.milestone2.analysis.toxicity_analysis import ToxicityAnalysis
from src.milestone2.actions.translate import Translator
from src.milestone2.actions.detoxify import Detoxifier

class Agent:
    def __init__(self, llm_name="gpt-4"):
        SYSTEM_PROMPT = """
            You are an AI assistant that analyzes and processes text using tools.

            You can use the following tools:
            - TranslateToEnglish: Translate a sentence into English. If the original entence is already in English, the "translated_text" field should simply be the original sentence.
            - SentimentAnalysisWithReason: Analyze the sentiment of a sentence and explain the reasoning.
            - ToxicityAnalysisWithReason: Analyze the toxicity of a sentence and explain the reasoning.
            - DetoxifySentence: Detoxify a sentence by removing harmful language and rewriting it. This tool should only be called once
            - FinalizeResponse: When you have completed your reasoning, you MUST call the FinalizeResponse tool to return your final answer.


            Once you have completed all tasks and gathered all necessary information, you MUST call FinalizeResponse.

            IMPORTANT: Do NOT output the final JSON in Thought.

            You MUST follow this exact format and ONLY return the JSON output:
            Thought: I have completed all reasoning and will now return the final answer.
            Action: FinalizeResponse
            Action Input: 
            {{
                "original_text": "...",
                "translated_text": "...",
                "sentiment_label": "...",
                "sentiment_explanation": "...",
                "toxicity_label": "...",
                "toxicity_explanation": "...",
                "detoxified_text": "..."
            }}

            ### Example Final ACTION
            Thought: I have completed the analysis and will now return the final answer.
            Action: FinalizeResponse
            Action Input: 
            {{
                "original_text": "==my butthole==  i love buttholes. they taste mmm goood. i will never stop vandalizing wikipedia. u no why?  cuz u dont tell me wat to do. k bye i hate you",
                "translated_text": "==my butthole==  i love buttholes. they taste mmm goood. i will never stop vandalizing wikipedia. u no why?  cuz u dont tell me wat to do. k bye i hate you",
                "sentiment_label": "Negative",
                "sentiment_explanation": "The overall sentiment of the sentence leans heavily towards negative due to explicit expressions of hate and advocacy of vandalism, overshadowing the initial light-heartedness.",
                "toxicity_label": "Toxic",
                "toxicity_explanation": "The sentence is considered toxic due to inappropriate language, references to vandalism, a confrontational tone, and expressions of hatred, which create a harmful and unwelcoming message.",
                "detoxified_text": "I appreciate various topics, but I prefer to maintain a respectful tone. Wiki editing may not be my priority, as I value personal autonomy. Farewell, and please understand that I do not have any negative feelings towards you."
            }}

            ### Example Final ANSWER
            {{
                "original_text": "Original sentence here.",
                "translated_text": "Translated sentence or original sentence if no translation is necessary.",
                "sentiment_label": "Positive/Negative/Neutral",
                "sentiment_explanation": "Explanation of the sentiment label.",
                "toxicity_label": "Toxic/Non-Toxic",
                "toxicity_explanation": "Explanation of the toxicity label.",
                "detoxified_text": "Detoxified version of the sentence if applicable."
            }}

            ### Important Notes
            - Your final response MUST contain all of the JSON attributes listed in the Example Final Answer. 
            - You MUST use the FinalizeResponse tool to return your final answer as a JSON object.
            - After performing translation, sentiment analysis, toxicity analysis, and detoxification (if required), you should finalize by calling FinalizeResponse.
            - Once you call FinalizeResponse, do not perform any more actions.
        """
        
        self.llm = LLM().create(key=llm_name)
        self.sentiment_analyzer = SentimentAnalysis(llm=self.llm)
        self.toxicity_analyzer = ToxicityAnalysis(llm=self.llm)
        self.translator = Translator()
        self.detoxifier = Detoxifier()
        
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
            ),
            Tool.from_function(
                func=self.finalize_response,
                name="FinalizeResponse",
                description="Use this tool ONLY when you have completed all reasoning and gathered all information. This MUST be your final action. Provide a complete JSON object as the Action Input.",
                return_direct=True
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={
                "prefix": SYSTEM_PROMPT
            },
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )

    def sentiment_analysis_with_reason(self, text: str) -> str:
        print("\n-----Performing sentiment analysis-----\n")
        return self.sentiment_analyzer.analyze(text)

    def toxicity_analysis_with_reason(self, text: str) -> str:
        print("\n-----Performing toxicity analysis-----\n")
        return self.toxicity_analyzer.analyze(text)

    def translate_to_english(self, text: str) -> str:
        print("\n-----Performing translation-----\n")
        try:
            return self.translator.translate(text)
        except Exception as e:
            print(f"Translation failed: {e}")
            # Return the required empty JSON structure through finalize_response
            return self.finalize_response({
                "original_text": text,
                "translated_text": "",
                "sentiment_label": "",
                "sentiment_explanation": "",
                "toxicity_label": "",
                "toxicity_explanation": "",
                "detoxified_text": ""
            })
    
    def detoxify_sentence(self, text: str) -> str:
        print("\n-----Performing detoxification-----\n")
        return self.detoxifier.detoxify(text)

    def finalize_response(self, json_output: dict):
        print("\n-----Final JSON Response-----\n")
        return json_output

    def run(self, text):
        response = self.agent.invoke({ "input": text })
        return response
