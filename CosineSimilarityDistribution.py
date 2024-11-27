#!/usr/bin/env python
# coding: utf-8

# In[ ]:


rom google.colab import drive
drive.mount('/content/drive')


# In[ ]:


from transformers import BartTokenizer, BartForConditionalGeneration

# Loading the tokenizer
tokenizer = BartTokenizer.from_pretrained('/content/drive/MyDrive/bart_tokenizer')

# Loading the model
model = BartForConditionalGeneration.from_pretrained('/content/drive/MyDrive/bart_model')


# In[ ]:


import pandas as pd

# Load the file
file_path = '/content/drive/MyDrive/Colab Notebooks/updated_text_analysis.csv'
data = pd.read_csv(file_path)


# In[ ]:


# Sample 20,000 records from the dataset
sampled_data = data.sample(n=100, random_state=42)


# In[ ]:


# Select the 'document' column
documents = sampled_data['document'].dropna().tolist()                           # Dropping any NaN values if present


# In[ ]:


def generate_summary(text):
    try:
        # Tokenize the input with truncation to handle long text
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary with controlled parameters
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode and return the summary text
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        print(f"Error generating summary for text: {text[:100]}...")             # Print a snippet of the text
        print(f"Error message: {e}")
        return None


# In[ ]:


from tqdm import tqdm

# Initialize an empty list to store summaries
all_summaries = []

# Define batch size
batch_size = 10

# Process documents in batches with a progress bar
for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
    # Get the current batch of documents
    batch = documents[i:i + batch_size]

    # Generate summaries for the batch
    batch_summaries = [generate_summary(text) for text in batch]

    # Append the batch summaries to the main list
    all_summaries.extend(batch_summaries)

# Add the summaries to the DataFrame
sampled_data['predicted_summary'] = all_summaries


# In[ ]:


# Save the results to a new CSV file in Google Drive
output_path = '/content/drive/MyDrive/dataset_with_summaries.csv'
sampled_data.to_csv(output_path, index=False)

print("Summaries saved successfully.")


# In[ ]:


pip install rake_nltk


# In[ ]:


import pandas as pd
from rake_nltk import Rake
import nltk

# Load the file
file_path = '/content/drive/MyDrive/dataset_with_summaries.csv'
sampled_data = pd.read_csv(file_path)

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')

sampled_data['predicted_summary'] = sampled_data['predicted_summary'].fillna("")
sampled_data['predicted_summary'] = sampled_data['predicted_summary'].astype(str)

# Initialize RAKE
rake = Rake()

# Function to extract keywords
def extract_top_keywords(text):
    rake.extract_keywords_from_text(text)                                        # Extract keywords from the text
    ranked_phrases = rake.get_ranked_phrases()                                   # Get keywords without scores
    return ranked_phrases[:3]                                                    # Return top 3 keywords

# Step 5: Apply RAKE to the 'document' column
sampled_data['top_keywords'] = sampled_data['predicted_summary'].apply(extract_top_keywords)

# Display the first few rows with the top keywords
sampled_data[['document', 'predicted_summary', 'top_keywords']].head()


# In[ ]:


sampled_data.info()


# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess the documents
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stop words
    words = [word for word in text.split() if word not in stop_words]
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Apply preprocessing to each document
cleaned_data = [preprocess_text(doc) for doc in sampled_data['document']]

# Convert the cleaned documents into a bag-of-words model
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(cleaned_data)

# Fit the LDA model
n_topics = 4
lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
lda.fit(doc_term_matrix)

# Display the topics and their top words
def display_topics(model, feature_names, no_top_words):
    for idx, topic in enumerate(model.components_):
        print(f"Topic {idx + 1}:")
        print([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        print("\n")

# Assign topic labels to each document
sampled_data['topic_label'] = lda.transform(doc_term_matrix).argmax(axis=1)

no_top_words = 10  # Number of words to display per topic

# Extract topic words
topic_words = []
for topic_idx, topic in enumerate(lda.components_):
    top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    topic_words.append(" ".join(top_words))

# Save the topic words in a new column by mapping topic labels to topic words
sampled_data['topic_words'] = sampled_data['topic_label'].map(lambda x: topic_words[x])

# Display the DataFrame with original text, topic label, and topic words
sampled_data[['document', 'topic_label', 'topic_words']]


# In[ ]:


# Get the count of each topic
topic_counts = sampled_data['topic_label'].value_counts()

# Display the topic counts
print(topic_counts)


# In[ ]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Function to calculate sentiment score
def get_sentiment_score(text):
    # Get the sentiment score for the text
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']  # Compound score is a general sentiment indicator

# Calculate sentiment scores for the columns
sampled_data['document_sentiment'] = sampled_data['document'].apply(get_sentiment_score)
sampled_data['summary_sentiment'] = sampled_data['summary'].apply(get_sentiment_score)
sampled_data['predicted_summary_sentiment'] = sampled_data['predicted_summary'].apply(get_sentiment_score)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the seaborn style for a clean look
sns.set(style="whitegrid")

# Custom color palette
colors = ['#66c2a5', '#fc8d62', '#8da0cb']

# Melting the DataFrame for use in seaborn
melted_data = sampled_data.melt(value_vars=['predicted_summary_sentiment', 'document_sentiment', 'summary_sentiment'],
                                var_name='Type', value_name='Sentiment')

# Creating a figure
plt.figure(figsize=(10, 6))

# KDE plot for density distribution
sns.kdeplot(data=melted_data[melted_data['Type'] == 'predicted_summary_sentiment'],
            x='Sentiment', color=colors[0], fill=True, label='Predicted Summary Sentiment', alpha=0.6)
sns.kdeplot(data=melted_data[melted_data['Type'] == 'document_sentiment'],
            x='Sentiment', color=colors[1], fill=True, label='Document Sentiment', alpha=0.6)
sns.kdeplot(data=melted_data[melted_data['Type'] == 'summary_sentiment'],
            x='Sentiment', color=colors[2], fill=True, label='Summary Sentiment', alpha=0.6)

# Title and labels with custom font sizes
plt.title('Density Distribution of Sentiments for Predicted Summary, Document, and Summary',
          fontsize=16, weight='bold')
plt.xlabel('Sentiment Polarity', fontsize=14)
plt.ylabel('Density', fontsize=14)

# Adding legend
plt.legend()

# Adjusting x-axis limits for better visualization
plt.xlim(-1, 1)

# Displaying the plot
plt.show()


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder

# Step 1: Text Preprocessing and Vectorization
# Define a function to preprocess and vectorize text
def preprocess_and_vectorize(data, column_name):
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    doc_term_matrix = vectorizer.fit_transform(data[column_name])
    return vectorizer, doc_term_matrix

# Vectorize 'document' and 'summary' columns
doc_vectorizer, doc_term_matrix = preprocess_and_vectorize(sampled_data, 'document')
summary_vectorizer, summary_term_matrix = preprocess_and_vectorize(sampled_data, 'summary')

# Step 2: Run LDA
# Define a function to fit the LDA model and get topics
def run_lda(term_matrix, n_topics=5):
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(term_matrix)
    return lda_model

# Run LDA on 'document' and 'summary' term matrices
n_topics = 4  # Define the number of topics you want
doc_lda = run_lda(doc_term_matrix, n_topics=n_topics)
summary_lda = run_lda(summary_term_matrix, n_topics=n_topics)

# Step 3: Assign Topics and Get Topic Labels
# Define a function to get the most likely topic for each text
def assign_topics(lda_model, term_matrix):
    topic_assignments = lda_model.transform(term_matrix)
    assigned_topics = topic_assignments.argmax(axis=1)
    return assigned_topics

# Assign topics to documents and summaries
sampled_data['document_topic'] = assign_topics(doc_lda, doc_term_matrix)
sampled_data['summary_topic'] = assign_topics(summary_lda, summary_term_matrix)

# Optional: Generate topic labels based on top words in each topic
def get_topic_labels(lda_model, vectorizer, n_top_words=5):
    words = vectorizer.get_feature_names_out()
    topic_labels = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [words[i] for i in topic.argsort()[-n_top_words:]][::-1]
        label = " ".join(top_words)
        topic_labels.append(label)
    return topic_labels

# Get topic labels for document and summary topics
document_topic_labels = get_topic_labels(doc_lda, doc_vectorizer)
summary_topic_labels = get_topic_labels(summary_lda, summary_vectorizer)

# Map topic numbers to labels
sampled_data['document_topic_label'] = sampled_data['document_topic'].apply(lambda x: document_topic_labels[x])
sampled_data['summary_topic_label'] = sampled_data['summary_topic'].apply(lambda x: summary_topic_labels[x])

# Display the results
sampled_data[['document', 'document_topic', 'document_topic_label',
                    'summary', 'summary_topic', 'summary_topic_label']].head()


# In[ ]:


sampled_data.info()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Function to compute cosine similarity between two text vectors
def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]

# Vectorize the documents, summaries, and predicted summaries (text columns only)
document_vectors = vectorizer.fit_transform(sampled_data['document'])
summary_vectors = vectorizer.transform(sampled_data['summary'])
predicted_summary_vectors = vectorizer.transform(sampled_data['predicted_summary'])

# Calculate and store cosine similarity for each pair
similarities_doc_summary = []
similarities_summary_predicted = []
similarities_doc_predicted = []

for idx, row in sampled_data.iterrows():
    # Get vector for the current document, summary, and predicted summary
    doc_vec = document_vectors[idx].toarray().flatten()
    summ_vec = summary_vectors[idx].toarray().flatten()
    pred_summ_vec = predicted_summary_vectors[idx].toarray().flatten()

    # Compute cosine similarities and append to lists
    similarities_doc_summary.append(compute_cosine_similarity(doc_vec, summ_vec))
    similarities_summary_predicted.append(compute_cosine_similarity(summ_vec, pred_summ_vec))
    similarities_doc_predicted.append(compute_cosine_similarity(doc_vec, pred_summ_vec))

# Add the new columns to the DataFrame
sampled_data['cosine_sim_doc_summary'] = similarities_doc_summary
sampled_data['cosine_sim_summary_predicted'] = similarities_summary_predicted
sampled_data['cosine_sim_doc_predicted'] = similarities_doc_predicted

# Check the resulting DataFrame
sampled_data[['cosine_sim_doc_summary', 'cosine_sim_summary_predicted', 'cosine_sim_doc_predicted']].head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.histplot(sampled_data['cosine_sim_doc_summary'], color="#1f77b4", label="Document vs Summary", kde=True)  # Blue
sns.histplot(sampled_data['cosine_sim_summary_predicted'], color="#ff7f0e", label="Summary vs Predicted Summary", kde=True)  # Orange
sns.histplot(sampled_data['cosine_sim_doc_predicted'], color="#2ca02c", label="Document vs Predicted Summary", kde=True)  # Green
plt.legend()
plt.xlabel("Cosine Similarity")
plt.title("Distribution of Cosine Similarities")
plt.show()


# In[ ]:




