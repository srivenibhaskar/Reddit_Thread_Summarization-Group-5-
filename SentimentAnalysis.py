#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


df = pd.read_csv('/content/drive/MyDrive/Project/cleaned_data.csv')

df.head(2)


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to calculate sentiment scores
def vader_sentiment(text):
    return sia.polarity_scores(text)

# Fill NaN values in lemmatized columns with empty strings
df['lemmatized_document'] = df['lemmatized_document'].fillna('')
df['lemmatized_summary'] = df['lemmatized_summary'].fillna('')

# Apply VADER sentiment analysis to document_cleaned and summary_cleaned columns
df['document_sentiment'] = df['lemmatized_document'].apply(lambda x: vader_sentiment(x)['compound'])
df['summary_sentiment'] = df['lemmatized_summary'].apply(lambda x: vader_sentiment(x)['compound'])

# Display the DataFrame with the sentiment scores
df[['lemmatized_document', 'document_sentiment', 'lemmatized_summary', 'summary_sentiment']].head(5)


# In[ ]:


# Group by sentiment scores for documents
grouped_document_sentiment = df.groupby('document_sentiment').size().reset_index(name='count')
grouped_document_sentiment['mean_summary_sentiment'] = df.groupby('document_sentiment')['summary_sentiment'].mean().values

# Group by sentiment scores for summaries
grouped_summary_sentiment = df.groupby('summary_sentiment').size().reset_index(name='count')
grouped_summary_sentiment['mean_document_sentiment'] = df.groupby('summary_sentiment')['document_sentiment'].mean().values

# Display the grouped DataFrames
print("Grouped Document Sentiment:")
print(grouped_document_sentiment)

print("\nGrouped Summary Sentiment:")
print(grouped_summary_sentiment)


# In[ ]:


def interpret_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['document_sentiment_category'] = df['document_sentiment'].apply(interpret_sentiment)
df['summary_sentiment_category'] = df['summary_sentiment'].apply(interpret_sentiment)

df[['lemmatized_document', 'document_sentiment_category', 'lemmatized_summary', 'summary_sentiment_category']].head(5)


# In[ ]:


import re

# Function to extract post/comment type and date from the ID
def extract_info(id_str):
    # Regular expression to capture the necessary components
    match = re.match(r'TLDR_(RS|RC)_(\d{4}-\d{2})-cm-\d+\.json', id_str)
    if match:
        return match.groups()
    return None, None

# Apply the extraction function to the 'id' column
df[['type', 'date']] = df['id'].apply(lambda x: pd.Series(extract_info(x)))
df.head(3)


# In[ ]:




