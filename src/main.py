### 1. Translation (as needed)
### 2. Preprocessing
### 3. Sentiment Analysis
### 4. Toxicity
### 5. Transfer Toxicity (as needed)

from src.analysis.chain_analysis import analyze_text

text = "I love the new features of this product, but sometimes it crashes."

res = analyze_text(text)
print(res)