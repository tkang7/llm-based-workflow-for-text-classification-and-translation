import pandas as pd
import ast
from .sentiment_and_toxicity_analysis import analyze_text
from ..processing.translate import translate_text

def analyze_and_classify(input_df, output_df, name):
    for index, row in input_df.iterrows():
        # translate if needed
        text = translate_text(row['sentence'])
        print(f"Index: {index}")
        print(f"Analyzing... : {text}")
        
        sentiment_result, toxicity_result = analyze_text(text)
        sentiment_result = ast.literal_eval(sentiment_result)
        toxicity_result = ast.literal_eval(toxicity_result)

        new_row = {
            "text": text,
            "sentiment_result": sentiment_result["sentiment_label"],
            "sentiment_explanation": sentiment_result["explanation"],
            "toxicity_result": toxicity_result["toxicity_label"],
            "toxicity_explanation": toxicity_result["explanation"]
        }
        output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
    
    print("Analysis complete, Exporting to CSV...")
    export_data_path = "./src/milestone1/data/output/" + name + ".csv"
    output_df.to_csv(export_data_path, index=False)
    print("Export complete!")

    return output_df
