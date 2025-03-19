from tqdm import tqdm
import pandas as pd
import json

from src.milestone2.models.agent import Agent
from src.milestone2.analysis.evaluate import Evaluate

class Main:
    def __init__(self):
        self.setup_agent()
        self.setup_input_data()
        self.setup_output_data()
        self.evaluate = Evaluate()

    def setup_agent(self):
        prod_llm = "mistralai/Mistral-7B-v0.1"
        # test_llm = "gpt-4"
        # gpt_mini = "gpt-4o-mini"
        # olama_qwen = "ollama-qwen:7b"
        # olama_llama = "ollama-llama2:7b-chat"
        # mistal_instruct = "ollama-mistral:7b-instruct"
        self.agent = Agent(llm_name=prod_llm) 

    def setup_input_data(self):
        input_data_path = "./src/milestone2/data/input/"
        milestone_2_detox_data_path = input_data_path + "Milestone-2-toxic-test-solutions.csv"
        milestone_2_multiligual_data_path = input_data_path + "Milstone-2-multilingual-sentiment-test-solutions.csv"
        milestone_1_sentiment_test_data_path = input_data_path + "eng_sentiment_test_solutions.csv"
        milestone_1_toxicity_test_data_path = input_data_path + "eng_toxicity_test-solutions.csv"

        self.input_data = {
            "milestone_1_sentiment_data": milestone_1_sentiment_test_data_path, # Completed Evaluation
            "milestone_1_toxicity_data": milestone_1_toxicity_test_data_path, # Completed Evaluation
            "milestone_2_detox_data": milestone_2_detox_data_path, # Completed Evaluation
            "milestone_2_multiligual_data": milestone_2_multiligual_data_path # Completed Evaluation
        }

    def setup_output_data(self):
        columns = ["original_text", "translated_text", "detoxified_text", "sentiment_result", "sentiment_explanation", "toxicity_result", "toxicity_explanation"]

        sentiment_test_predictions = pd.DataFrame(columns=columns)
        toxicity_test_predictions = pd.DataFrame(columns=columns)
        detoxify_test_predictions = pd.DataFrame(columns=columns)
        multilingual_test_predictions = pd.DataFrame(columns=columns)

        self.output_data = {
            "milestone_1_sentiment_data": sentiment_test_predictions,
            "milestone_1_toxicity_data": toxicity_test_predictions,
            "milestone_2_detox_data": detoxify_test_predictions,    
            "milestone_2_multiligual_data": multilingual_test_predictions
        }

    def analyze_and_export(self, input_df, output_df, name):
        for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing Rows"):
            text = row["sentence"]

            print(f"Index: {index}")
            print(f"Analyzing... : {text}\n")
            processed_text = self.agent.run(text)
            print(f"\nFinished Analyzing Text of type {type(processed_text)}: {processed_text}\n")
            
            if isinstance(processed_text, dict):
                raw_output = processed_text["output"]
                json_result = json.loads(raw_output)
            else:
                if "```json" in processed_text:
                    processed_text = processed_text.split("```json")[-1].strip()
                    processed_text = processed_text.strip("```").strip()
                
                json_result = json.loads(processed_text, strict=False)

            new_row = {
                "original_text": json_result["original_text"],
                "translated_text": json_result["translated_text"],
                "detoxified_text": json_result["detoxified_text"],
                "sentiment_result": json_result["sentiment_label"],
                "sentiment_explanation": json_result["sentiment_explanation"],
                "toxicity_result": json_result["toxicity_label"],
                "toxicity_explanation": json_result["toxicity_explanation"]
            }
            output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
        
        print("Analysis complete, Exporting to CSV...")
        export_data_path = "./src/milestone2/data/output/" + name + ".csv"
        output_df.to_csv(export_data_path, index=False)
        print("Export complete!")

        return output_df
    
    def run(self):
        for name, input_path in self.input_data.items():
            # if name.startswith("milestone_1"): continue # skipping milestone1
            input = pd.read_csv(input_path)
            output = self.output_data[name]
            print("Analyzing Data: " + name)
            result_df = self.analyze_and_export(input, output, name)
            print("Finished Analayzing Data: " + name)

            print("Evaluating Data: " + name)
            self.evaluate.evaluate(result_df, name)
            print("Finished Evaluating Data: " + name)

if __name__ == "__main__":
    ### TEST CODE
    # test_sentence = "ይህ የኮምፒዩተር መጨመሪያ ወዲያውኑ ይሰራል፣ ግን በሌላ ቦታ መያዝ አልቻልኩም።" 
    # main = Main()
    # main.agent.run(test_sentence)
    ### TEST CODE

    ### PROD CODE
    main = Main()
    main.run()
    ### PROD CODE
