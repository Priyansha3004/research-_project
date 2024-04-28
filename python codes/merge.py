import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the CSV file into DataFrame
df = pd.read_csv("ai.csv")

# Remove rows with missing content
df = df.dropna(subset=['Content'])

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the content into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(df['Content'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Visualize the cosine similarity matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Cosine Similarity Matrix')
plt.xlabel('Document Index')
plt.ylabel('Document Index')
plt.show()
