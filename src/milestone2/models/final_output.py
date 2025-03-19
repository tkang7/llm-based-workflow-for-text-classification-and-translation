from pydantic import BaseModel, Field

class FinalOutput(BaseModel):
    original_text: str = Field(..., description="The original input sentence")
    translated_text: str = Field(..., description="The sentence translated into English, if applicable")
    sentiment_label: str = Field(..., description="The sentiment label for the sentence, e.g., Positive, Negative, Neutral")
    sentiment_explanation: str = Field(..., description="An explanation justifying the sentiment label")
    toxicity_label: str = Field(..., description="The toxicity label for the sentence, e.g., Toxic, Non-Toxic")
    toxicity_explanation: str = Field(..., description="An explanation justifying the toxicity label")
    detoxified_text: str = Field(..., description="The detoxified version of the sentence")