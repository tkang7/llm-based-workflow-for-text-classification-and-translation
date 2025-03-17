import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

actual_data_path = "./src/milestone1/data/input/"
types_of_data = ["sentiment", "toxicity"]

def evaluate(pred, name):
    actual=""
    name = name.lower()

    if name in types_of_data:
        if name == types_of_data[0]:
            sentiment_actual_data_path = actual_data_path + "eng_sentiment_test_solutions.csv"
            sentiment_test_actual = pd.read_csv(sentiment_actual_data_path)
            actual = sentiment_test_actual["class-label"].str.lower()
        else:
            toxicity_actual_data_path = actual_data_path + "eng_toxicity_test-solutions.csv"
            toxicity_test_actual = pd.read_csv(toxicity_actual_data_path)
            actual = toxicity_test_actual["label"].str.lower()
    else:
        print(f"THIS TYPE OF EVALUATION IS NOT SUPPORTED: {name}")

    pred = pred.str.lower()

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

    # Export metrics to a text file
    with open(f"./src/milestone1/data/output/{name}_evaluation.txt", "w") as file:
        file.write(f"Evaluating {name}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    pred_data_path = "./src/milestone1/data/output/"
    sentiment_test_data_path = pred_data_path + "sentiment_test_predictions.csv"
    toxicity_test_data_path = pred_data_path + "toxicity_test_predictions.csv"
    sentiment_test_prediction = pd.read_csv(sentiment_test_data_path)
    toxicity_test_prediction = pd.read_csv(toxicity_test_data_path)

    # === Evaluate Sentiment ===
    evaluate(
        pred=sentiment_test_prediction["sentiment_result"],
        name="Sentiment"
    )

    # === Evaluate Toxicity ===
    evaluate(
        pred=toxicity_test_prediction["toxicity_result"],
        name="Toxicity"
    )