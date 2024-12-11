#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, ArrayType, DoubleType
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName("TextAnalysis").getOrCreate()


# In[ ]:


df = spark.read.option("header", "true").csv('dbfs:/FileStore/cleaned_data.csv')
df.show(5)


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Fill NaN values in 'document' and 'summary'
df = df.fillna({'document': '', 'summary': ''})

# VADER Sentiment Analysis
sia = SentimentIntensityAnalyzer()

# Define UDF for sentiment analysis
def vader_sentiment_udf(text):
    score = sia.polarity_scores(text)['compound']
    return score

# Register UDF
vader_udf = udf(vader_sentiment_udf, DoubleType())

# Apply the UDF to compute sentiment scores
df = df.withColumn("document_sentiment", vader_udf(col("document")))
df = df.withColumn("summary_sentiment", vader_udf(col("summary")))


# In[ ]:


# Interpret sentiment scores
def interpret_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Register interpret sentiment as UDF
interpret_udf = udf(interpret_sentiment, StringType())

df = df.withColumn("document_sentiment", interpret_udf(col("document_sentiment")))
df = df.withColumn("summary_sentiment", interpret_udf(col("summary_sentiment")))

df[['document', 'document_sentiment', 'summary', 'summary_sentiment']].show(5)


# In[ ]:


from gensim import corpora
from gensim.models import LdaModel

# Tokenization
def tokenize(text):
    return text.split() if isinstance(text, str) else []

tokenize_udf = udf(tokenize, ArrayType(StringType()))

df = df.withColumn("document_token", tokenize_udf(col("document")))
df = df.withColumn("summary_token", tokenize_udf(col("summary")))

# Topic Modeling
def perform_lda(tokens_list):
    if tokens_list:
        dictionary = corpora.Dictionary([tokens_list])
        corpus = [dictionary.doc2bow(tokens_list)]
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, random_state=42)
        topic_words = lda_model.show_topic(0, topn=5)
        return " + ".join([word for word, _ in topic_words])
    return "No Topic"

lda_udf = udf(perform_lda, StringType())

df = df.withColumn("document_topics", lda_udf(col("document_token")))
df = df.withColumn("summary_topics", lda_udf(col("summary_token")))


# In[ ]:


df[['document', 'document_topics', 'summary', 'summary_topics']].show(5)


# In[ ]:


from pyspark.sql.functions import concat_ws

# Convert array columns to string columns
df = df.withColumn("document_token", concat_ws(" ", col("document_token")))
df = df.withColumn("summary_token", concat_ws(" ", col("summary_token")))


# In[ ]:


# Output path for the CSV file
output_path = "dbfs:/FileStore/text_analysed_data.csv"

# Write the DataFrame to a CSV file
df.write.option("header", "true").csv(output_path)


# In[ ]:




