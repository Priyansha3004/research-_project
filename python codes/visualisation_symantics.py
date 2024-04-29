import pandas as pd
import spacy
import seaborn as sns
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Read the CSV files with specified data types or set low_memory=False
ai_df = pd.read_csv("ai.csv", dtype=str)
human_df = pd.read_csv("human.csv", dtype=str)

# Process AI-generated content
ai_errors = []
for index, row in ai_df.iterrows():
    ai_sentence = str(row["Content"])  # Convert to string
    ai_doc = nlp(ai_sentence)
    ai_errors.append(len([token for token in ai_doc if token.is_oov]))

# Process human-generated content
human_errors = []
for index, row in human_df.iterrows():
    human_sentence = str(row["Content"])  # Convert to string
    human_doc = nlp(human_sentence)
    human_errors.append(len([token for token in human_doc if token.is_oov]))

# Ensure lengths of both lists are the same
min_length = min(len(ai_errors), len(human_errors))
ai_errors = ai_errors[:min_length]
human_errors = human_errors[:min_length]

# Create a DataFrame for heatmap
data = pd.DataFrame({"AI Errors": ai_errors, "Human Errors": human_errors})

# Create a heatmap
sns.heatmap(data, annot=True, fmt="d")
plt.title("Semantic Error Heatmap")
plt.xlabel("Content Type")
plt.ylabel("Samples")
plt.show()
