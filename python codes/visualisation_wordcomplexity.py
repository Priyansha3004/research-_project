import pandas as pd
import matplotlib.pyplot as plt
from textstat import syllable_count  # Import syllable_count directly from textstat

# Read the CSV files into DataFrames with explicit data type specification and low_memory=False
ai_df = pd.read_csv("ai.csv", dtype=str, usecols=['Content'], low_memory=False)
human_df = pd.read_csv("human.csv", dtype=str, usecols=['Content'], low_memory=False)

# Define a function to calculate Flesch Reading Ease score based on word complexity
def calculate_flesch_reading_ease(text):
    words = text.split()  # Split the text into words
    word_count = len(words)  # Count the number of words
    syllable_count_total = sum([syllable_count(word) for word in words])  # Count syllables in each word and sum them up
    if word_count == 0:
        return 0
    return 206.835 - 1.015 * (word_count / 1.0) - 84.6 * (syllable_count_total / word_count)  # Calculate Flesch Reading Ease score

# Calculate Flesch Reading Ease score for AI-generated content
ai_df['Flesch_Reading_Ease_AI'] = ai_df['Content'].apply(lambda x: calculate_flesch_reading_ease(str(x)))

# Calculate Flesch Reading Ease score for human-generated content
human_df['Flesch_Reading_Ease_Human'] = human_df['Content'].apply(lambda x: calculate_flesch_reading_ease(str(x)))

# Plot the distribution of Flesch Reading Ease scores for AI and human content
plt.figure(figsize=(10, 6))
plt.hist(ai_df['Flesch_Reading_Ease_AI'].astype(float), bins=20, alpha=0.5, label='AI-generated Content')
plt.hist(human_df['Flesch_Reading_Ease_Human'].astype(float), bins=20, alpha=0.5, label='Human-generated Content')
plt.xlabel('Flesch Reading Ease Score')
plt.ylabel('Frequency')
plt.title('Distribution of Word Complexity')
plt.legend()
plt.grid(True)
plt.show()
