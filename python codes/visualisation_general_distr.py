import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the CSV files into DataFrames
ai_df = pd.read_csv("aigenerated.csv")
human_df = pd.read_csv("human.csv")

# Replace missing values with an empty string
ai_df['Content'] = ai_df['Content'].fillna('')
human_df['Content'] = human_df['Content'].fillna('')

# Preprocess text (e.g., remove punctuation, convert to lowercase)
ai_df['Content'] = ai_df['Content'].str.lower().replace(r'[^\w\s]', '').str.strip()
human_df['Content'] = human_df['Content'].str.lower().replace(r'[^\w\s]', '').str.strip()

# Combine AI-generated and human-generated content
combined_content = ai_df['Content'].tolist() + human_df['Content'].tolist()

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform TF-IDF vectorizer on combined content
tfidf_matrix = vectorizer.fit_transform(combined_content)

# Calculate cosine similarity between AI-generated and human-generated content
similarity_matrix = cosine_similarity(tfidf_matrix)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix[:len(ai_df), len(ai_df):], cmap="coolwarm", cbar=True, square=True, xticklabels=human_df.index, yticklabels=ai_df.index)
plt.title("Plagiarism Heatmap: AI vs Human Generated Content")
plt.xlabel("Human Generated Content")
plt.ylabel("AI Generated Content")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
