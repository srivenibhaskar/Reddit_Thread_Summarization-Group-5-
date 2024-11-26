#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# loading the dataset and displaying the first five records
df = pd.read_csv(r"C:\Users\sushi\Downloads\merged_data.csv")
df.head()


# In[5]:


df.info()


# In[7]:


# converting columns to appropriate datatypes
df = df.astype({
    'document': 'string',  
    'summary': 'string',   
    'id': 'category',      
    'document_sentiment': 'float64',  
    'summary_sentiment': 'float64', 
    'document_sentiment_category': 'category',  
    'summary_sentiment_category': 'category',   
    'document_topics': 'category', 
    'summary_topics': 'category'
})

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])


# In[9]:


df.info()


# In[11]:


# counting the number of words in 'tokenized_document'
df['word_count_document'] = df['tokenized_document'].apply(len)
df[['document', 'word_count_document']].head()


# In[13]:


# counting the number of words in 'tokenized_summary'
df['word_count_summary'] = df['tokenized_summary'].apply(len)
df[['summary', 'word_count_summary']].head()


# In[15]:


# calculate the proportion of the number of words in the document with respect to the summary
df['document_to_summary_word_proportion'] = df['word_count_document'] / df['word_count_summary']

# handle any potential division by zero if summaries are empty
df['document_to_summary_word_proportion'] = df['document_to_summary_word_proportion'].fillna(0)  # Set to 0 if summary word count is 0
df[['document', 'word_count_document', 'summary', 'word_count_summary', 'document_to_summary_word_proportion']].head()


# In[17]:


# Function to count sentences
def count_sentences(text):
    # Check if the text is not NaN and is a string
    if isinstance(text, str):
        # Split by '.', '!', and '?' and filter out any empty strings
        return len([s for s in text.split('.') + text.split('!') + text.split('?') if s.strip()])
    return 0  # Return 0 for NaN or non-string entries

# Count the number of sentences in each document
df['no_of_sentences_document'] = df['document'].apply(count_sentences)

# Count the number of sentences in each summary
df['no_of_sentences_summary'] = df['summary'].apply(count_sentences)

# Calculate the proportion of the number of sentences in the document with respect to the summary
df['document_to_summary_sentence_proportion'] = df['no_of_sentences_document'] / df['no_of_sentences_summary']

# Handle any potential division by zero if summaries have no sentences
df['document_to_summary_sentence_proportion'] = df['document_to_summary_sentence_proportion'].fillna(0)  # Set to 0 if summary sentence count is 0
df[['document', 'no_of_sentences_document', 'summary', 'no_of_sentences_summary', 'document_to_summary_sentence_proportion']].head()


# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the required columns exist in the DataFrame
required_columns = [
    'word_count_document',
    'word_count_summary',
    'no_of_sentences_document',
    'no_of_sentences_summary'
]

# Check for missing columns
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Missing columns in the DataFrame: {missing_columns}")
else:
    # Set the style for the plots
    sns.set(style="whitegrid")

    # Create a figure to hold the subplots
    plt.figure(figsize=(18, 6))

    # Scatter plot for Number of Words in Document vs. Number of Words in Summary
    plt.subplot(1, 3, 2)
    sns.scatterplot(x='word_count_document', y='word_count_summary', data=df)
    plt.title('Number of Words in Document vs. Number of Words in Summary')
    plt.xlabel('Number of Words in Document')
    plt.ylabel('Number of Words in Summary')
    plt.xlim(0, df['word_count_document'].max() + 10)  # Adjust the limits as needed
    plt.ylim(0, df['word_count_summary'].max() + 10)
    plt.axhline(0, color='grey', lw=0.8, ls='--')  # Add horizontal line at y=0
    plt.axvline(0, color='grey', lw=0.8, ls='--')  # Add vertical line at x=0

    # Scatter plot for Number of Sentences in Document vs. Number of Sentences in Summary
    plt.subplot(1, 3, 3)
    sns.scatterplot(x='no_of_sentences_document', y='no_of_sentences_summary', data=df)
    plt.title('Number of Sentences in Document vs. Number of Sentences in Summary')
    plt.xlabel('Number of Sentences in Document')
    plt.ylabel('Number of Sentences in Summary')
    plt.xlim(0, df['no_of_sentences_document'].max() + 1)  # Adjust the limits as needed
    plt.ylim(0, df['no_of_sentences_summary'].max() + 1)
    plt.axhline(0, color='grey', lw=0.8, ls='--')  # Add horizontal line at y=0
    plt.axvline(0, color='grey', lw=0.8, ls='--')  # Add vertical line at x=0

    # Adjust the layout to prevent overlap
    plt.tight_layout()
    plt.show()


# In[21]:


# create a contingency matrix
matrix = pd.crosstab(df['document_sentiment_category'], df['summary_sentiment_category'], 
                     rownames=['sentiment_summary'], 
                     colnames=['document_summary'], 
                     dropna=False)

# creating a heatmap
plt.figure(figsize=(10, 6)) 
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=True, square=True)
plt.title('Contingency Matrix: Document vs Summary Sentiment Categories', fontsize=16)
plt.xlabel('Document Sentiment Category', fontsize=12)
plt.ylabel('Summary Sentiment Category', fontsize=12)
plt.show()


# In[23]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoders for both columns
le_summary_topics = LabelEncoder()
le_document_topics = LabelEncoder()

# Fit and transform the 'summary_topics' and 'document_topics' columns
df['summary_topics_encoded'] = le_summary_topics.fit_transform(df['summary_topics'])
df['document_topics_encoded'] = le_document_topics.fit_transform(df['document_topics'])
df[['summary_topics', 'summary_topics_encoded', 'document_topics', 'document_topics_encoded']].head()


# In[25]:


df = df.astype({
    'summary_topics_encoded': 'category',  
    'document_topics_encoded': 'category'
})


# In[27]:


# computing Pearson and Spearman correlations between document_topics and summary_topics
pearson_corr = df['document_topics_encoded'].corr(df['summary_topics_encoded'], method='pearson')
spearman_corr = df['document_topics_encoded'].corr(df['summary_topics_encoded'], method='spearman')

print(f"Pearson Correlation: {pearson_corr}")
print(f"Spearman Correlation: {spearman_corr}")

if abs(pearson_corr) > 0.5:
    print("Strong positive/negative linear relationship between document and summary topics.")
else:
    print("Weak or no linear relationship between document and summary topics.")


# In[31]:


# creating a heatmap between document_topics and summary_topics
matrix = pd.crosstab(df['document_topics_encoded'], df['summary_topics_encoded'], 
                     rownames=['summary_topics'], 
                     colnames=['document_topics'], 
                     dropna=False)

# creating a heatmap
plt.figure(figsize=(12, 8))  # Set the figure size
sns.heatmap(matrix, annot=True, fmt='d', cmap='coolwarm', cbar=True, square=True)
plt.title('Contingency Matrix: Document vs Summary Topics', fontsize=16)
plt.xlabel('Document Topics', fontsize=12)
plt.ylabel('Summary Topics', fontsize=12)
plt.show()


# # Cosine Similarity

# In[42]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Convert columns to arrays
doc_topic_vectors = df[['document_topics_encoded']].values
summary_topic_vectors = df[['summary_topics_encoded']].values

# Initialize an empty list to store similarity scores
similarities = []

# Compute similarity for each pair row-by-row
for i in range(len(df)):
    similarity = cosine_similarity([doc_topic_vectors[i]], [summary_topic_vectors[i]])[0][0]
    similarities.append(similarity)

# Add the similarity scores to the DataFrame
df['topic_similarity'] = similarities

# Display a few results
df[['document_topics_encoded', 'summary_topics_encoded', 'topic_similarity']].head()


# In[44]:


# group by 'topic_similarity' and counting the number of records
topic_counts = df.groupby('topic_similarity').size().reset_index(name='count')
print(topic_counts)


# In[48]:


# dropping the specified columns
columns_to_drop = [
    'id',
    'type',
    'removed_document_stopwords',
    'removed_summary_stopwords',
    'word_count_document',
    'word_count_summary',
    'no_of_sentences_document',
    'no_of_sentences_summary',
    'summary_topics',
    'document_topics',
    'topic_similarity'
]

df.drop(columns=columns_to_drop, inplace=True)
df.info()


# In[50]:


# encoding categorical features
label_encoders = {}
categorical_cols = ['document_sentiment_category', 'summary_sentiment_category', 
                    'summary_topics_encoded', 'document_topics_encoded']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    # store encoder if needed for inverse transformation
    label_encoders[col] = le

# select only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64', 'category'])

# compute the correlation matrix
correlation_matrix = numeric_df.corr(method='pearson')

# visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix', fontsize=16)
plt.show()

