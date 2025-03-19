import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Evaluate:
    INPUT_DATA_PATH = "./src/milestone2/data/input/"
    INPUT_DATA_CATEGORIES = ["milestone_1_sentiment_data", "milestone_1_toxicity_data", "milestone_2_detox_data", "milestone_2_multiligual_data"]
    
    def __init__(self):
        pass

    def evaluate(self, pred_df, name):
        actual=""
        name = name.lower()

        if name in self.INPUT_DATA_CATEGORIES:
            if name == self.INPUT_DATA_CATEGORIES[0]:
                sentiment_actual_data_path = self.INPUT_DATA_PATH + "eng_sentiment_test_solutions.csv"
                sentiment_test_actual = pd.read_csv(sentiment_actual_data_path)
                actual = sentiment_test_actual["class-label"].str.lower()
                pred = pred_df["sentiment_result"]
            elif name == self.INPUT_DATA_CATEGORIES[1]:
                toxicity_actual_data_path = self.INPUT_DATA_CATEGORIES + "eng_toxicity_test-solutions.csv"
                toxicity_test_actual = pd.read_csv(toxicity_actual_data_path)
                actual = toxicity_test_actual["label"].str.lower()
                pred = pred_df["toxicity_result"]
            elif name == self.INPUT_DATA_CATEGORIES[2]:
                detox_actual_data_path = self.INPUT_DATA_PATH + "Milestone-2-toxic-test-solutions.csv"
                detox_test_actual = pd.read_csv(detox_actual_data_path)
                actual = detox_test_actual["source_label"].str.lower()
                pred = pred_df["toxicity_result"]
            else:
                detox_actual_data_path = self.INPUT_DATA_PATH + "Milstone-2-multilingual-sentiment-test-solutions.csv"
                detox_test_actual = pd.read_csv(detox_actual_data_path)
                actual = detox_test_actual["class-label"].str.lower()
                print(pred_df)
                pred = pred_df["sentiment_result"].str.lower()
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
        with open(f"./src/milestone2/data/output/{name}_evaluation.txt", "w") as file:
            file.write(f"Evaluating {name}\n")
            file.write(f"Accuracy: {accuracy:.4f}\n")
            file.write(f"Precision: {precision:.4f}\n")
            file.write(f"Recall: {recall:.4f}\n")
            file.write(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    pred_data_path = "./src/milestone2/data/output/"
    sentiment_test_data_path = pred_data_path + "milestone_2_multiligual_data.csv"
    sentiment_test_prediction = pd.read_csv(sentiment_test_data_path)

    evaluate = Evaluate()

    # === Evaluate Sentiment ===
    evaluate.evaluate(
        sentiment_test_prediction,
        "milestone_2_multiligual_data"
    )