### sentiment analysis
from langchain.chat_models import ChatOpenAI

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
            response = self.llm.invoke(prompt)
        else:
            # self.llm is a transformers pipeline
            outputs = self.llm(prompt, max_length=256, do_sample=True)
            
            response = outputs[0]['generated_text']
            
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
        return response