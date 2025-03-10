from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from langchain import HuggingFacePipeline
from . import sentiment as sen
from . import toxicity as tox
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# api_key = os.getenv("OPENAI_API_KEY")

device = "cpu"
model_name = "ibm-granite/granite-3.0-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map=device)
model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_length=512,   # You can tweak this higher if needed
)

# sequential_chain is no longer used because we switched to HuggingFace pipeline
# sequential_chain = SequentialChain(
#     chains=[sen.sentiment_chain(llm), tox.toxicity_chain(llm)],
#     input_variables=["text"], 
#     output_variables=["sentiment_result", "toxicity_result"],
#     verbose=True
# )

def analyze_text(text):
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)

    sentiment_chain = sen.sentiment_chain(hf_pipeline)
    toxicity_chain = tox.toxicity_chain(hf_pipeline)

    sentiment_output = sentiment_chain({"text": text})
    toxicity_output = toxicity_chain({"text": text})

    return (
        sentiment_output["sentiment_result"],
        toxicity_output["toxicity_result"]
    )

# def analyze_text(text):
#     sentiment_result = sen.sentiment_chain(pipe)(text)
#     toxicity_result = tox.toxicity_chain(pipe)(text)
#     return (sentiment_result, toxicity_result)