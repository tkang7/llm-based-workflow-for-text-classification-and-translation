### LLM instatiation
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.llms import OpenAI  # Switched from ChatOpenAI
from langchain_ollama import ChatOllama
import torch
import os

class LLM:
    def __init__(self):
        self.keys = {
            "small": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "medium": "ibm-granite/granite-3.0-2b-instruct",
            "gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
            "ollama-qwen:7b": "qwen:7b",
            "ollama-llama3.2": "llama3.2",
            "gpt-3.5-turbo-instruct": "gpt-3.5-turbo-instruct",  # OpenAI completion model
            "gpt-4o": "gpt-4o",
            "text-davinci-003": "text-davinci-003"  # Another OpenAI completion model
        }

    def create(self, key="small"):
        if key.lower().startswith("gpt"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required for OpenAI models")

            return OpenAI(
                model=self.keys[key],
                temperature=0,
                openai_api_key=api_key
            )
        
        if key.lower().startswith("ollama"):
            return ChatOllama(
                model=self.keys[key],
                temperature=0,
            )
        
        model_id = self.keys[key.lower()] # select model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1,
            max_length=512,
            temperature=0
        )

        return generator
    
    def empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        else:
            return