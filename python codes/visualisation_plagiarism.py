import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the CSV files into DataFrames with explicit data type specification and low_memory=False
ai_df = pd.read_csv("aigenerated.csv", dtype=str, low_memory=False)
human_df = pd.read_csv("human.csv", dtype=str, low_memory=False)

# Combine AI-generated and human-generated content, handling NaN values
combined_content = (ai_df['Content'].fillna('') + ' ' + human_df['Content'].fillna('')).astype(str)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=10000)

# Fit and transform TF-IDF vectorizer on combined content
tfidf_matrix = vectorizer.fit_transform(combined_content)

# Separate AI and human content
ai_tfidf = tfidf_matrix[:len(ai_df)]
human_tfidf = tfidf_matrix[len(ai_df):]

# Calculate cosine similarity
similarities = cosine_similarity(ai_tfidf, human_tfidf)

# Print mean similarity
print("Mean Cosine Similarity:", similarities.mean())

# Flatten the similarities matrix to get a 1D array of all similarity scores
flat_similarities = similarities.flatten()

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(flat_similarities, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Cosine Similarity Scores')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
