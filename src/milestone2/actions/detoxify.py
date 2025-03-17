from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class Detoxifier():
    def __init__(self):
        model_name = "UBC-NLP/DetoxLLM-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.detoxify_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256,
            temperature=0.1
        )

    def detoxify(self, text):
        prompt = (
            "### Instruction:\n"
            "Detoxify the following sentence while keeping its meaning intact.\n\n"
            f"### Input:\n{text}\n\n### Response:"
        )
        response = self.detoxify_pipeline(prompt, num_return_sequences=1)
        detoxified_text = response[0]['generated_text'].split("### Response:")[-1].strip()
        
        return detoxified_text