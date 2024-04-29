import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textstat import lexicon_count

# Read the CSV files
ai_df = pd.read_csv("ai.csv", dtype=str)
human_df = pd.read_csv("human.csv", dtype=str)

# Calculate vocabulary diversity for AI-generated content
ai_vocabulary = ai_df["Content"].apply(lambda x: lexicon_count(str(x)))

# Calculate vocabulary diversity for human-generated content
human_vocabulary = human_df["Content"].apply(lambda x: lexicon_count(str(x)))

# Create a DataFrame for vocabulary diversity comparison
vocabulary_df = pd.DataFrame({"AI Vocabulary": ai_vocabulary, "Human Vocabulary": human_vocabulary})

# Visualize using different methods
# Method 1: Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=vocabulary_df)
plt.title("Vocabulary Diversity Comparison")
plt.ylabel("Vocabulary Count")
plt.xlabel("Content Type")
plt.xticks(rotation=45)
plt.show()

# Method 2: Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=vocabulary_df, bins=20, kde=True, alpha=0.5)
plt.title("Vocabulary Diversity Distribution")
plt.xlabel("Vocabulary Count")
plt.ylabel("Frequency")
plt.legend(labels=["AI", "Human"])
plt.show()

# Method 3: Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x="AI Vocabulary", y="Human Vocabulary", data=vocabulary_df)
plt.title("Vocabulary Diversity Scatter Plot")
plt.xlabel("AI Vocabulary")
plt.ylabel("Human Vocabulary")
plt.show()
