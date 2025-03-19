### LLM instatiation
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_ollama import ChatOllama
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import login
import torch
import os

class LLM:
    def __init__(self):
        self.keys = {
            "small": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "medium": "ibm-granite/granite-3.0-2b-instruct",
            "mistralai/Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
            "ollama-mistral:7b-instruct": "mistral:7b-instruct",
            "gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini": "gpt-4o-mini",
            "ollama-qwen:7b": "qwen:7b",
            "ollama-llama3.2": "llama3.2",
            "ollama-llama2:7b-chat": "llama2:7b-chat",
            "gpt-3.5-turbo-instruct": "gpt-3.5-turbo-instruct", 
            "gpt-4o": "gpt-4o",
            "gpt-4": "gpt-4",
            "text-davinci-003": "text-davinci-003"
        }
        self.llm_temperature = 0.5

    def create(self, key="small"):
        if key.lower().startswith("gpt"):
            api_key = os.getenv("OPENAI_API_KEY")
            print(api_key)
            if not api_key:
                raise ValueError("OpenAI API key required for OpenAI models")

            print(f"Created LLM using {key}")
            return ChatOpenAI(
                model_name=key,
                temperature=self.llm_temperature,
                openai_api_key=api_key,
                model_kwargs={"logit_bias": {}}
            )
        
        if key.lower().startswith("ollama"):
            print(f"Created LLM using {key}")
            return ChatOllama(
                model=self.keys[key],
                temperature=self.llm_temperature,
            )
        
        hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")
        login(hugging_face_token)

        model_id = self.keys[key] # select model
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hugging_face_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_auth_token=hugging_face_token
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Created LLM Using {device}")
        model = model.to(device)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1,
            max_new_tokens=512,
            temperature=self.llm_temperature
        )

        print(f"Created LLM using {key}")
        return HuggingFacePipeline(pipeline=generator)
    
    def empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        else:
            return