import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

actual_data_path = "./data/input/"
types_of_data = ["sentiment", "toxicity"]

def evaluate(pred, name):
    actual=""

    if name.lower() in types_of_data:
        if name == types_of_data[0]:
            sentiment_actual_data_path = actual_data_path + "eng_sentiment_test_solutions.csv"
            sentiment_test_actual = pd.read_csv(sentiment_actual_data_path)
            actual=sentiment_test_actual["class-label"], 
            pred=pred["sentiment_result"],
        else:
            toxicity_actual_data_path = actual_data_path + "eng_sentiment_test_solutions.csv"
            toxicity_test_actual = pd.read_csv(toxicity_actual_data_path)
            actual=toxicity_test_actual["class-label"], 
            pred=pred["toxicity_result"],
    else:
        print(f"THIS TYPE OF EVALUATION IS NOT SUPPORTED: {name}")

    actual = actual.str.lower()
    pred = actual.str.lower()

    # Calculate metrics
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='macro')
    recall = recall_score(actual, pred, average='macro')
    f1 = f1_score(actual, pred, average='macro')

    print(f"Evaluating {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    sentiment_actual_data_path = actual_data_path + "eng_setiment_test_solutions.csv"
    toxicity_actual_data_path = actual_data_path + "eng_toxicity_test-solutions.csv"
    sentiment_test_actual = pd.read_csv(sentiment_actual_data_path)
    toxicity_test_actual = pd.read_csv(toxicity_actual_data_path)

    pred_data_path = "./data/output/"
    sentiment_test_data_path = pred_data_path + "sentiment_test_predictions.csv"
    toxicity_test_data_path = pred_data_path + "toxicity_test_predictions.csv"
    sentiment_test_prediction = pd.read_csv(sentiment_test_data_path)
    toxicity_test_prediction = pd.read_csv(toxicity_test_data_path)

    # === Evaluate Sentiment ===
    evaluate(
        actual=sentiment_test_actual["class-label"], 
        pred=sentiment_test_prediction["sentiment_result"],
        name="Sentiment Analysis"
    )

    # === Evaluate Toxicity ===
    evaluate(
        actual=toxicity_test_actual["label"], 
        pred=toxicity_test_prediction["toxicity_result"],
        name="Toxicity Analysis"
    )