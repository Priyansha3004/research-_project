import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textstat import flesch_reading_ease

# Read the CSV files
ai_df = pd.read_csv("ai.csv", dtype=str)
human_df = pd.read_csv("human.csv", dtype=str)

# Calculate readability scores for AI-generated content
ai_readability = ai_df["Content"].apply(lambda x: flesch_reading_ease(str(x)))

# Calculate readability scores for human-generated content
human_readability = human_df["Content"].apply(lambda x: flesch_reading_ease(str(x)))

# Create a DataFrame for readability comparison
readability_df = pd.DataFrame({"AI Readability": ai_readability, "Human Readability": human_readability})

# Visualize using different methods
# Method 1: Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=readability_df)
plt.title("Readability Comparison")
plt.ylabel("Readability Score")
plt.xlabel("Content Type")
plt.xticks(rotation=45)
plt.show()

# Method 2: Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=readability_df, bins=20, kde=True, alpha=0.5)
plt.title("Readability Distribution")
plt.xlabel("Readability Score")
plt.ylabel("Frequency")
plt.legend(labels=["AI", "Human"])
plt.show()

# Method 3: Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x="AI Readability", y="Human Readability", data=readability_df)
plt.title("Readability Scatter Plot")
plt.xlabel("AI Readability")
plt.ylabel("Human Readability")
plt.show()
