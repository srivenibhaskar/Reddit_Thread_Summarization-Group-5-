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

# In[ ]:




