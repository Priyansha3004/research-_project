import pandas as pd
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV file into DataFrame
df = pd.read_csv('finalmerge.csv')

# Preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

# Compute cosine similarity
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity

# Check plagiarism
plagiarism_threshold = 0.9  # Adjust threshold as needed
for index, row in df.iterrows():
    if 'generation' in row and row['generation'] == 'ai':
        ai_text = row['text']
        for idx, r in df.iterrows():
            if 'generation' in r and r['generation'] == 'human':
                human_text = r['text']
                similarity = compute_similarity(ai_text, human_text)
                if similarity > plagiarism_threshold:
                    print(f"Potential plagiarism found between AI text (index {index}) and human text (index {idx}). Similarity: {similarity}")

# Additional checks
for index, row in df.iterrows():
    if row.loc['generation'] == 'ai':
        ai_text = str(row.loc['text'])  # Convert ai_text to string
        human_texts = df[(df['generation'] == 'human') & (df['type'] == row.loc['type'])]['text']
        # Length comparison
        ai_length = len(ai_text)
        for human_text in human_texts:
            human_length = len(human_text)
            length_difference = abs(ai_length - human_length)
            if length_difference <= 50:  # Adjust threshold as needed
                print(f"Similar length found between AI text (index {index}) and human text. Length difference: {length_difference}")
        # Vocabulary usage
        ai_tokens = set(word_tokenize(ai_text))
        for human_text in human_texts:
            human_tokens = set(word_tokenize(human_text))
            common_tokens = ai_tokens.intersection(human_tokens)
            if len(common_tokens) >= 10:  # Adjust threshold as needed
                print(f"Common vocabulary found between AI text (index {index}) and human text.")
