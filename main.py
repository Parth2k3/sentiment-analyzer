import os
import pandas as pd
import string
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load dataset
df = pd.read_csv('user_review.csv')

# Data Cleaning
df.dropna(subset=['review'], inplace=True)  # Remove rows with null reviews
if 'unnecessary_column' in df.columns:
    df.drop(columns=['unnecessary_column'], inplace=True)  # Drop unnecessary columns

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Sentiment Analysis with VADER
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    vs = analyzer.polarity_scores(text)
    if vs['compound'] >= 0.4:
        return 'positive'
    elif vs['compound'] <= -0.1:
        return 'negative'
    else:
        return 'neutral'

df['vader_sentiment'] = df['cleaned_review'].apply(analyze_sentiment_vader)
df.to_csv('cleaned_user_reviews.csv', index=False)

#PLotting the results
vader_sentiment_counts = df['vader_sentiment'].value_counts()
vader_sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('VADER Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.savefig('vader_sentiment_distribution.png')
plt.show()
