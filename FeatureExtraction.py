#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install sentence_transformers


# In[ ]:


pip install textstat


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#importing the datastet
df = pd.read_csv('/content/cleaned_data.csv')
df.head(5)


# In[ ]:


# Check for NaN values in the entire DataFrame
nan_counts = df.isnull().sum()

# Display the number of NaN values for each column
print(nan_counts)


# In[ ]:


# Display the current column names
print(df.columns)



# In[ ]:


# Rename the last two columns
df.rename(columns={df.columns[-2]: 'document_cleaned', df.columns[-1]: 'summary_cleaned'}, inplace=True)

# Verify the changes
df.head(5)


# In[ ]:


# Using vectorized operations for sentence length calculation
df['Cleaned_document_length'] = df['document_cleaned'].str.split().str.len()
df['Cleaned_summary_length'] = df['summary_cleaned'].str.split().str.len()

df['document_length'] = df['document'].str.split().str.len()  # Length for document
df['summary_length'] = df['summary'].str.split().str.len()

# Selecting the required columns in the desired order
columns_to_display = [
    'document', 'summary', 'id', 'rg_labels', 'ext_labels',
    'document_cleaned', 'summary_cleaned', 'document_length','summary_length','Cleaned_document_length', 'Cleaned_summary_length'
]

# Creating a new DataFrame with the selected columns
df_formatted = df[columns_to_display]

# Option 1: Display in Jupyter Notebook
df_formatted.head(5)  # Display the DataFrame


# In[ ]:


# Set the ROUGE threshold (you can adjust this value)
threshold = 0.6

# Calculate the average ROUGE score for each list in rg_labels
df['avg_rg_score'] = df['rg_labels'].apply(lambda x: sum(x) / len(x) if isinstance(x, list) else float('nan'))

# Classify summaries as 'abstractive' or 'extractive' based on the average ROUGE score
df['summary_type'] = df['avg_rg_score'].apply(lambda x: 'abstractive' if x < threshold else 'extractive')

# Calculate the value counts for the summary types
summary_type_counts = df['summary_type'].value_counts(normalize=True)

# Get the proportions for both 'abstractive' and 'extractive' summaries
abstractive_proportion = summary_type_counts.get('abstractive', 0)
extractive_proportion = summary_type_counts.get('extractive', 0)

print(f"Proportion of abstractive summaries: {abstractive_proportion:.2%}")
print(f"Proportion of extractive summaries: {extractive_proportion:.2%}")


# In[ ]:


# Set the ROUGE threshold (you can adjust this value)
threshold = 0.5

# Calculate the average ROUGE score for each list in rg_labels
df['avg_rg_score'] = df['rg_labels'].apply(lambda x: sum(x)/len(x) if isinstance(x, list) else float('nan'))

# Classify summaries as 'abstractive' or 'extractive' based on the average ROUGE score
df['summary_type'] = df['avg_rg_score'].apply(lambda x: 'abstractive' if x < threshold else 'extractive')

# Check if 'abstractive' exists in the summary_type column before calculating the proportion
if 'abstractive' in df['summary_type'].value_counts(normalize=True):
    abstractive_proportion = df['summary_type'].value_counts(normalize=True)['abstractive']
    print(f"Proportion of abstractive summaries: {abstractive_proportion:.2%}")
else:
    print("No abstractive summaries found.")


# In[ ]:


plt.figure(figsize=(12, 6))

# Histogram for Cleaned Document Length
sns.histplot(df['Cleaned_document_length'], bins=30, color='blue', label='Document_c Length', kde=True, alpha=0.5)
# Histogram for Cleaned Summary Length
sns.histplot(df['Cleaned_summary_length'], bins=30, color='orange', label='Summary_c Length', kde=True, alpha=0.5)

plt.title('Distribution of Document and Summary Lengths')
plt.xlabel('Length (Number of Words)')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


# caluculation the propotion of extractive summary from the documnet using the
# Calculate the proportion of extractive summary
df['Proportion_extractive_summary(%)'] = (df['Cleaned_summary_length'] / df['Cleaned_document_length']) *100

# Display the updated DataFrame with the new column
df.head(3)


# In[ ]:


plt.figure(figsize=(8,6))
plt.plot(df.index, df['Cleaned_document_length'], label='Document Length', marker='o')
plt.plot(df.index, df['Proportion_extractive_summary(%)'], label='Proportion of Extractive Summary', marker='o')
plt.xlabel('Documents')
plt.ylabel('Value')
plt.title('Document Length vs Proportion of Extractive Summary')
plt.legend()
plt.show()


# In[ ]:


# Find the maximum proportion of extractive summary
max_proportion = df['Proportion_extractive_summary(%)'].max()
min_proportion = df['Proportion_extractive_summary(%)'].min()
print('max_prop',max_proportion)
print('min_prop',min_proportion)


# **Interpretation**:
# The plot shows a clear distinction between document lengths and their summaries. While document lengths vary widely, the proportion of the summary extracted is relatively small and stable.
# There is a visible outlier in the proportion of extractive summary near the 20,000th document where the proportion spikes slightly higher than the rest.

# In[ ]:


# Check for NaN values in the entire DataFrame
nan_counts = df.isnull().sum()

# Display the number of NaN values for each column
print(nan_counts)


# In[ ]:


# Filling  NaN values with empty strings
df['document_cleaned'].fillna('', inplace=True)
df['summary_cleaned'].fillna('', inplace=True)


# In[ ]:


import nltk
from nltk.corpus import words

# Download the words corpus
nltk.download('words')

# Create a set of valid words
valid_words = set(words.words())

# Function to remove gibberish words and return as a single string
def remove_gibberish(text):
    return ' '.join([word for word in text.split() if word.lower() in valid_words])

# Apply this to your cleaned document
df['document_without_gibberish'] = df['document_cleaned'].apply(remove_gibberish)
df['summary_without_gibberish'] = df['summary_cleaned'].apply(remove_gibberish)


# In[ ]:


df.head(5)


# In[ ]:


# Apply TFIDF
tfidf_vectorizer_document = TfidfVectorizer()
tfidf_vectorizer_summary = TfidfVectorizer()

tfidf_document = tfidf_vectorizer_document.fit_transform(df['document_without_gibberish'])
tfidf_summary = tfidf_vectorizer_summary.fit_transform(df['summary_without_gibberish'])


# In[ ]:


tfidf_document_df = pd.DataFrame.sparse.from_spmatrix(tfidf_document, columns=tfidf_vectorizer_document.get_feature_names_out())
tfidf_summary_df = pd.DataFrame.sparse.from_spmatrix(tfidf_summary, columns=tfidf_vectorizer_summary.get_feature_names_out())


# In[ ]:


#display
display(tfidf_document_df.head())


# In[ ]:


display(tfidf_summary_df.head())


# In[ ]:


# Displaying the words with highest scores in TD-IDF
#Sum scores of each word for document
word_score_document = tfidf_document_df.sum().sort_values(ascending=False)


# In[ ]:


print("Words with highest scores in TF-IDF for Documents")
print(word_score_document.head())


# In[ ]:


#Sum scores of each word for Summary
word_score_summary = tfidf_summary_df.sum().sort_values(ascending=False)


# In[ ]:


print("Words with highest scores in TF-IDF for summary")
print(word_score_summary.head())


# # Readability Score

# In[ ]:


# Function to calculate Flesch Reading Ease
def calculate_flesch_reading_ease(text):
    if pd.isnull(text):
        return None
    return textstat.flesch_reading_ease(text)

# Apply the Flesch Reading Ease function to the document and summary columns
df['Flesch_Reading_Ease_doc'] = df['document'].apply(calculate_flesch_reading_ease)
df['Flesch_Reading_Ease_Summary'] = df['summary'].apply(calculate_flesch_reading_ease)

# Display the updated DataFrame with Flesch Reading Ease scores
df_readability = df[['document', 'summary', 'Flesch_Reading_Ease_doc', 'Flesch_Reading_Ease_Summary']]
df_readability.head(5)


# # Flesch Reading Ease: A higher score indicates easier readability.

# In[ ]:


# Sample a subset of the DataFrame (e.g., the first 10 rows)
df_sample = df.head(10)  # Adjust the number of rows as needed

# Plotting
plt.figure(figsize=(10, 6))

# Set bar width
bar_width = 0.35

# Set positions of bar on X axis
r1 = range(len(df_sample))
r2 = [x + bar_width for x in r1]

# Create bars
plt.bar(r1, df_sample['Flesch_Reading_Ease_doc'], color='b', width=bar_width, edgecolor='grey', label='Document')
plt.bar(r2, df_sample['Flesch_Reading_Ease_Summary'], color='g', width=bar_width, edgecolor='grey', label='Summary')

# Add labels
plt.xlabel('Documents', fontweight='bold', fontsize=15)
plt.xticks([r + bar_width / 2 for r in range(len(df_sample))], [f'Doc {i+1}' for i in range(len(df_sample))])  # x-axis labels
plt.ylabel('Flesch Reading Ease Score', fontweight='bold', fontsize=15)
plt.title('Comparison of Flesch Reading Ease Score: Document vs. Summary', fontweight='bold', fontsize=16)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


# **Summaries are easier to read: The summaries (green bars) have higher readability scores compared to the original documents (blue bars).**
# 
# **Complexity of original documents: Many of the original documents have low or negative Flesch Reading Ease Scores, indicating that they are difficult to read.**
# 
# **Consistent improvement: In every case, the summaries simplify the content, making it more readable.**
# 
# **Notable differences: Some documents (e.g., Doc 3, Doc 5, Doc 7) have very low or negative scores, but their summaries show significant improvements in readability.**

# In[ ]:


# Rename the columns in the DataFrame
df.rename(columns={'document_cleaned': 'Lemmatized_document',
                   'summary_cleaned': 'lemmatized_summary'}, inplace=True)

# Display the updated DataFrame to confirm the changes
df.head()


# In[ ]:


# Remove the 'document length' and 'summary length' columns
df.drop(columns=['document_length', 'summary_length'], inplace=True)

# Display the updated DataFrame to confirm the changes
df.head()


# In[ ]:




