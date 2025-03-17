from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def detoxify(input):
    model_name = "UBC-NLP/DetoxLLM-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    prompt = "Rewrite the following toxic input into non-toxic version. Let's break the input down step by step to rewrite the non-toxic version. You should first think about the expanation of why the input text is toxic. Then generate the detoxic output. You must preserve the original meaning as much as possible.\nInput: "

    input = "Those shithead should stop talking and get the f*ck out of this place"
    input_text = prompt+input+"\n"

    print("Tokenizing Input Text...\n")
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    print("Finished Tokenizing Input Text...\n")

    print("Encoding Input Text...\n")
    outputs = model.generate(**input_ids, do_sample=False)
    print("Finished Encoding Input Text...\n")

    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    input = "Those shithead should stop talking and get the f*ck out of this place"

    detoxify(input)
