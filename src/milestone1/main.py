import pandas as pd
from src.milestone1.analysis.analyze import analyze_and_classify
from src.milestone1.evaluation.base_evaluation import evaluate

def run_analysis_and_evaluation(input_df, output_df, name_df):
    result = analyze_and_classify(input_df, output_df, name_df)
    name = ""

    if name_df == "sentiment_test_predictions":
        name = "sentiment"
        result = result["sentiment_result"]
    elif name_df == "toxicity_test_predictions":
        name = "toxicity"
        result = result["toxicity_result"]
    else:
        print(f"THIS TYPE OF EVALUATION IS NOT SUPPORTED: {name_df}")

    evaluate(result, name)

if __name__ == "__main__":
    input_data_path = "./src/milestone1/data/input/"
    sentiment_test_data_path = input_data_path + "eng_sentiment_test_solutions.csv"
    toxicity_test_data_path = input_data_path + "eng_toxicity_test-solutions.csv"

    sentiment_test_df = pd.read_csv(sentiment_test_data_path)
    toxicity_test_df = pd.read_csv(toxicity_test_data_path)

    sentiment_test_predictions = pd.DataFrame(columns=["text", "sentiment_result", "sentiment_explanation", "toxicity_result", "toxicity_explanation"])
    toxicity_test_predictions = pd.DataFrame(columns=["text", "sentiment_result", "sentiment_explanation", "toxicity_result", "toxicity_explanation"])

    run_analysis_and_evaluation(sentiment_test_df, sentiment_test_predictions, "sentiment_test_predictions")
    run_analysis_and_evaluation(toxicity_test_df, toxicity_test_predictions, "toxicity_test_predictions")