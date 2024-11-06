#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

from google.colab import drive
drive.mount('/content/drive')


# In[3]:


df = pd.read_csv('/content/drive/MyDrive/Project/text_analysis.csv')

df.head(2)


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

# Count the occurrences of each sentiment per topic for document and summary
doc_sentiment_distribution = df.groupby(['document_topics', 'document_sentiment_category']).size().unstack(fill_value=0)
summary_sentiment_distribution = df.groupby(['summary_topics', 'summary_sentiment_category']).size().unstack(fill_value=0)

# Setting up the plot with two subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Plot for document topics
doc_sentiment_distribution.plot(kind='bar', stacked=True, colormap='coolwarm', edgecolor='black', ax=axes[0])
axes[0].set_title('Sentiment Distribution by Document Topics', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Topics', fontsize=12)
axes[0].set_ylabel('Number of Documents', fontsize=12)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=10)
axes[0].legend(title='Sentiment', title_fontsize='13', fontsize='10')

# Plot for summary topics
summary_sentiment_distribution.plot(kind='bar', stacked=True, colormap='coolwarm', edgecolor='black', ax=axes[1])
axes[1].set_title('Sentiment Distribution by Summary Topics', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Topics', fontsize=12)
axes[1].set_ylabel('Number of Summaries', fontsize=12)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=10)
axes[1].legend(title='Sentiment', title_fontsize='13', fontsize='10')

# Styling for professional look
sns.set_style("whitegrid")
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# **Insight :**
# 
# 1) In both charts, certain topics (like - "one, like, time, feel") are much more prevalent in terms of the number of documents or summaries, and they tend to carry a significant amount of Negative sentiment.
# 
# 2) Neutral sentiment appears less frequent in general compared to Negative and Positive sentiments across both documents and summaries.

# In[5]:


# Create a figure for Document Lengths
plt.figure(figsize=(10, 5))
for sentiment in df['document_sentiment_category'].unique():
    sns.histplot(
        df[df['document_sentiment_category'] == sentiment]['document'].apply(len).dropna(),  # Drop NaN values
        bins=20,
        label=sentiment,
        kde=True,  # Kernel density estimate
        alpha=0.5
    )

plt.title('Length of Documents by Sentiment Category', fontsize=16, fontweight='bold')
plt.xlabel('Length of Document', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Sentiment Category', title_fontsize='13', fontsize='10')
sns.set_style("whitegrid")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# **Insight :**
# 
# This visualization may reflect the emotional engagement associated with different sentiment types - positive sentiments often inspire longer, more detailed repsonses, while negative or neutral opinions might require fewer words to express.

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Initialize CountVectorizer to create bigrams and filter out punctuation
vectorizer = CountVectorizer(
    ngram_range=(2, 2),
    token_pattern=r'(?u)\b\w+\b'  # This pattern captures words and ignores punctuation
)

# Step 2: Fit and transform the documents
X = vectorizer.fit_transform(df['document'])  # Replace 'document' with your actual column name

# Step 3: Get the bigram frequencies
bigram_freq = X.sum(axis=0).A1  # Sum up the counts for each bigram
bigram_names = vectorizer.get_feature_names_out()  # Get the bigram names

# Step 4: Create a DataFrame for visualization
bigram_freq_df = pd.DataFrame({'Bigram': bigram_names, 'Frequency': bigram_freq})
bigram_freq_df = bigram_freq_df.sort_values(by='Frequency', ascending=False).head(20)  # Top 20 bigrams

# Step 5: Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='Frequency', y='Bigram', data=bigram_freq_df, palette='viridis')
plt.title('Top 20 Bigrams in Documents', fontsize=16, fontweight='bold')
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Bigram', fontsize=12)
plt.grid(axis='x')
plt.tight_layout()
plt.show()


# **Insight :**
# 
# The N-gram frequency analysis provides insights into the linguistic patterns present in the collection of documents. The prevalence of specific bigrams, particularly "and i," "i am," and "in the," highlights common conversational and narrative structures within the text. These findings suggest a strong emphasis on personal expression, indicating that the documents likely reflect individual experiences and sentiments.

# In[ ]:




