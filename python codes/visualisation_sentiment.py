import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Read the CSV files
ai_df = pd.read_csv("ai.csv", dtype=str)
human_df = pd.read_csv("human.csv", dtype=str)

# Perform sentiment analysis for AI-generated content
ai_sentiment = ai_df["Content"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Perform sentiment analysis for human-generated content
human_sentiment = human_df["Content"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Create a DataFrame for sentiment analysis comparison
sentiment_df = pd.DataFrame({"AI Sentiment": ai_sentiment, "Human Sentiment": human_sentiment})

# Visualize using different methods
# Method 1: Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=sentiment_df)
plt.title("Sentiment Analysis Comparison")
plt.ylabel("Sentiment Polarity")
plt.xlabel("Content Type")
plt.xticks(rotation=45)
plt.show()

# Method 2: Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=sentiment_df, bins=20, kde=True, alpha=0.5)
plt.title("Sentiment Analysis Distribution")
plt.xlabel("Sentiment Polarity")
plt.ylabel("Frequency")
plt.legend(labels=["AI", "Human"])
plt.show()

# Method 3: Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x="AI Sentiment", y="Human Sentiment", data=sentiment_df)
plt.title("Sentiment Analysis Scatter Plot")
plt.xlabel("AI Sentiment")
plt.ylabel("Human Sentiment")
plt.show()
