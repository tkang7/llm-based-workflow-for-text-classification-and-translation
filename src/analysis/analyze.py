import pandas as pd
import numpy as np
import sys
import json
import os
from tqdm import tqdm
from .chain_analysis import analyze_text
from ..preprocessing.translate import translate_text

def analyze_and_classify(input_df, output_df, name):
    for _, row in tqdm(input_df.iterrows(), desc="Analyzing Sentiment Test Dataset"):
        # translate if needed
        text = translate_text(row['sentence'])
        
        sentiment_result, toxicity_result = analyze_text(text)
        sentiment_result = json.loads(sentiment_result)
        toxicity_result = json.loads(toxicity_result)

        new_row = {
            "text": text,
            "sentiment_result": sentiment_result["sentiment_label"],
            "sentiment_explanation": sentiment_result["explanation"],
            "toxicity_result": toxicity_result["toxicity_label"],
            "toxicity_explanation": toxicity_result["explanation"]
        }
        output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)

    # export to csv

    export_data_path = "./src/data/output/"
    output_df.to_csv(export_data_path + name + '.csv', index=False)

    return output_df
