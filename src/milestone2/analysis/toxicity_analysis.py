### sentiment analysis
from langchain.chat_models import ChatOpenAI

class ToxicityAnalysis:
    def __init__(self, llm=None):
        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    def analyze(self, text, use_openAI=True):
        prompt = (
            f"Is the following sentence toxic or non-toxic? "
            f"Explain your reasoning in detail.\n\n"
            f"Sentence: \"{text}\""
        )

        if use_openAI:
            response = self.llm.invoke(prompt)
        else:
            # self.llm is a transformers pipeline
            outputs = self.llm(prompt, max_new_tokens=300, do_sample=True)
            
            response = outputs[0]['generated_text']
            
            # If the prompt is repeated in the response, trim it:
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
        return response