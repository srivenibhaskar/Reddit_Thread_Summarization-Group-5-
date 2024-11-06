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


df.columns


# In[ ]:


from gensim import corpora
from gensim.models import LdaModel

# Ensure each entry in 'lemmatized_document' and 'lemmatized_summary' is a list of tokens
# For example, you might need to split the strings into lists if they are currently strings
df['lemmatized_document'] = df['lemmatized_document'].apply(lambda x: x.split() if isinstance(x, str) else x)
df['lemmatized_summary'] = df['lemmatized_summary'].apply(lambda x: x.split() if isinstance(x, str) else x)

# Create a dictionary and corpus for the lemmatized documents
dictionary_docs = corpora.Dictionary(df['lemmatized_document'])
corpus_docs = [dictionary_docs.doc2bow(text) for text in df['lemmatized_document']]

# Create and fit the LDA model for the lemmatized documents
lda_model_docs = LdaModel(corpus=corpus_docs, id2word=dictionary_docs, num_topics=10, random_state=42)

# Get the topics for the lemmatized documents
topics_docs = [lda_model_docs[doc] for doc in corpus_docs]

# Add the dominant topic for each document to the DataFrame
df['document_topics'] = [max(topic, key=lambda x: x[1])[0] if topic else -1 for topic in topics_docs]

# Create a dictionary and corpus for the lemmatized summaries
dictionary_summaries = corpora.Dictionary(df['lemmatized_summary'])
corpus_summaries = [dictionary_summaries.doc2bow(text) for text in df['lemmatized_summary']]

# Create and fit the LDA model for the lemmatized summaries
lda_model_summaries = LdaModel(corpus=corpus_summaries, id2word=dictionary_summaries, num_topics=10, random_state=42)

# Get the topics for the lemmatized summaries
topics_summaries = [lda_model_summaries[doc] for doc in corpus_summaries]

# Add the dominant topic for each summary to the DataFrame
df['summary_topics'] = [max(topic, key=lambda x: x[1])[0] if topic else -1 for topic in topics_summaries]


# In[ ]:


df[['lemmatized_document', 'document_topics', 'lemmatized_summary', 'summary_topics']].head(5)


# In[ ]:


import string

# Function to clean and limit the topics
def clean_topics(topics, num_keywords=5):
    cleaned_topics = {}

    for topic_num, words in topics.items():

        # Remove punctuation, quotes, and filter out empty strings
        filtered_words = [word.strip(string.punctuation).lower() for word in words if word.strip(string.punctuation).isalpha()]

        # Limit to the first 'num_keywords' words
        cleaned_topics[topic_num] = filtered_words[:num_keywords]

    return cleaned_topics

# Clean the topics
cleaned_doc_topic_keywords = clean_topics(doc_topic_keywords)

# Print the cleaned topics
print("Cleaned Document Topics and their Keywords:")
for topic_num, words in cleaned_doc_topic_keywords.items():
    print(f"Topic {topic_num}: {', '.join(words) if words else 'No valid words'}")


# In[ ]:


topic_mapping = {
    0: "dave, pod, alex, jeff, pierce",
    1: "game, one, nt, player, get",
    2: "people, would, one, nt, like",
    3: "hundred, two, thousand, one, three",
    4: "one, nt, like, get, time",
    5: "nt, like, time, know, feel",
    6: "share, price, short, stock, market",
    7: "sex, woman, men, girl, sexual",
    8: "car, nt, one, would, work",
    9: "nt, get, would, know, post"
}

# Replace the topic numbers with corresponding keywords
df['document_topics'] = df['document_topics'].map(topic_mapping)
df['summary_topics'] = df['summary_topics'].map(topic_mapping)


# In[ ]:


from nltk import pos_tag, word_tokenize

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define the POS tagging function
def pos_tagging(text):
    if isinstance(text, str):  # Check if input is a string
        tokens = word_tokenize(text)
        return [(token, pos) for token, pos in pos_tag(tokens)]
    else:
        return []  # Return an empty list for non-string inputs

# Apply POS tagging to both columns
df['document_pos'] = df['document'].apply(pos_tagging)
df['summary_pos'] = df['summary'].apply(pos_tagging)


# In[ ]:


df[['document_pos', 'summary_pos']]


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')

df.to_csv('/content/drive/MyDrive/Project/text_analysis.csv', index=False)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Project/text_analysis.csv')

df.head(2)


# In[ ]:


from nltk import pos_tag, word_tokenize, ne_chunk

nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to perform NER using NLTK
def ner_extraction(text):
    if isinstance(text, str):  # Check if input is a string
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)  # Get POS tags
        named_entities = ne_chunk(pos_tags)  # Perform NER

        # Extracting named entities
        return [(entity[0], entity.label()) for entity in named_entities if hasattr(entity, 'label')]
    else:
        return []  # Return an empty list for non-string inputs

# Apply NER to both columns
df['document_ner'] = df['document'].apply(ner_extraction)
df['summary_ner'] = df['summary'].apply(ner_extraction)


# In[ ]:


df[['document', 'document_ner', 'summary', 'summary_ner']]


# In[ ]:




