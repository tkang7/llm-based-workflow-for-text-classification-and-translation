### 1. Translation (as needed)
### 2. Preprocessing
### 3. Sentiment Analysis
### 4. Toxicity
### 5. Transfer Toxicity (as needed)

import pandas as pd
import numpy as np
from tqdm import tqdm
from src.analysis.analyze import analyze_and_classify
from src.evaluation.base_evaluation import evaluate

def run_analysis_and_evaluation(input_df, output_df, name_df):
    result = analyze_and_classify(input_df, output_df, name_df)
    name = ""

    if name_df == "sentiment_test_predictions":
        name = "sentiment"
    if name_df == "toxicity_test_predictions":
        name = "toxicity"
    else:
        print(f"THIS TYPE OF EVALUATION IS NOT SUPPORTED: {name_df}")

    evaluate(result, name)

if __name__ == "__main__":
    input_data_path = "./src/data/input/"
    sentiment_test_data_path = input_data_path + "eng_sentiment_test_solutions.csv"
    toxicity_test_data_path = input_data_path + "eng_toxicity_test-solutions.csv"

    sentiment_test_df = pd.read_csv(sentiment_test_data_path)
    toxicity_test_df = pd.read_csv(toxicity_test_data_path)

    sentiment_test_predictions = pd.DataFrame(columns=["text", "sentiment_result", "sentiment_explanation", "toxicity_result", "toxicity_explanation"])
    toxicity_test_predictions = pd.DataFrame(columns=["text", "sentiment_result", "sentiment_explanation", "toxicity_result", "toxicity_explanation"])

    run_analysis_and_evaluation(sentiment_test_df, sentiment_test_predictions, "sentiment_test_predictions")
    run_analysis_and_evaluation(toxicity_test_df, toxicity_test_predictions, "toxicity_test_predictions")

    