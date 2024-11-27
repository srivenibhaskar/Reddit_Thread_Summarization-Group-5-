#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf
from pyspark.sql.types import StringType
import re

# Initialize SparkSession
spark = SparkSession.builder.appName("DataCleaning").getOrCreate()


# In[ ]:


df1 = spark.read.json('dbfs:/FileStore/train_0.json')
df2 = spark.read.json('dbfs:/FileStore/train_1.json')

# Combine the DataFrames using union()
df = df1.union(df2)

df.show(5)


# In[ ]:


# Define UDF for removing special characters
def remove_special_characters(text):
    text = re.sub(r'http\S+|www\S+|@\S+|<.*?>', '', text)  # Remove HTML tags and URLs
    text = re.sub(r'\b\w+@\w+\.\w+\b', '', text)           # Remove email addresses
    text = re.sub(r'@\w+', '', text)                      # Remove usernames starting with '@'
    text = re.sub(r'\bu/\w+\b', '', text)                 # Remove Reddit usernames
    text = re.sub(r'\s*\(\s*', ' ', text)                 # Remove parentheses and spaces
    text = re.sub(r'\s*\)\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()              # Replace multiple spaces with one
    return text

remove_special_characters_udf = udf(remove_special_characters, StringType())


# In[ ]:


# Apply special character removal to document and summary columns
df = df.withColumn("document", remove_special_characters_udf(col("document")))
df = df.withColumn("summary", remove_special_characters_udf(col("summary")))


# In[ ]:


# Convert text to lowercase
df = df.withColumn("document", lower(col("document")))
df = df.withColumn("summary", lower(col("summary")))


# In[ ]:


# Comprehensive custom slang dictionary
slang_dictionary = {
    "u": "you",
    "r": "are",
    "cuz": "because",
    "dont": "do not",
    "wont": "will not",
    "im": "I am",
    "yall": "you all",
    "gonna": "going to",
    "gotta": "got to",
    "hafta": "have to",
    "lemme": "let me",
    "kinda": "kind of",
    "sorta": "sort of",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "btw": "by the way",
    "fyi": "for your information",
    "smh": "shaking my head",
    "idk": "I don't know",
    "ftw": "for the win",
    "brb": "be right back",
    "tbh": "to be honest",
    "wyd": "what you doing",
    "salty": "bitter or upset",
    "simp": "someone who shows excessive sympathy",
    "sus": "suspicious",
    "vibe check": "assessing someone's energy or mood",
    "lit": "exciting or excellent",
    "yeet": "to throw something with force",
    "ghosting": "sudden cut-off communication",
    "shook": "shocked or surprised",
    "extra": "over the top",
    "b4": "before",
    "gtg": "got to go",
    "omg": "oh my god",
    "imo": "in my opinion",
    "tldr": "too long; didn't read",
    "ikr": "I know right",
    "rofl": "rolling on the floor laughing",
    "yolo": "you only live once",
    "ama": "ask me anything",
    "asap": "as soon as possible",
    "nsfw": "not safe for work",
    "afaik": "as far as I know",
    "wtf": "what the f***",
    "irl": "in real life",
    "afk": "away from keyboard",
    "np": "no problem",
    "fr": "for real",
    "srsly": "seriously",
    "fam": "family",
    "flex": "show off",
    "shade": "disrespect",
    "clout": "influence or power",
    "cap/no cap": "lie/no lie",
    "stan": "an obsessive fan",
    "thirsty": "desperate for attention",
    "fomo": "fear of missing out",
    "bussin": "really good",
    "bet": "agreement or approval",
    "cheugy": "out of touch or trying too hard",
}

# Compile regex pattern for slang replacement
slang_pattern = r'\b(' + '|'.join(re.escape(slang) for slang in slang_dictionary.keys()) + r')\b'

# Define UDF for replacing slangs
def replace_slangs(text):
    return re.sub(slang_pattern, lambda x: slang_dictionary[x.group(0)], text)

replace_slangs_udf = udf(replace_slangs, StringType())


# In[ ]:


# Apply slang replacement
df = df.withColumn("document", replace_slangs_udf(col("document")))
df = df.withColumn("summary", replace_slangs_udf(col("summary")))

# Show the processed data
display(df.limit(5))


# In[ ]:


from pyspark.sql.functions import col, concat_ws

# Transform array columns to string
df = df.withColumn("ext_labels", concat_ws(",", col("ext_labels"))) \
    .withColumn("rg_labels", concat_ws(",", col("rg_labels")))


# In[ ]:


# Output path for the CSV file
output_path = "dbfs:/FileStore/cleaned_data.csv"

# Write the DataFrame to a CSV file
df.write.option("header", "true").csv(output_path)


# In[ ]:




