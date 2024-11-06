#!/usr/bin/env python
# coding: utf-8

# In[44]:


import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[45]:


folder_path = r"/content/Project"

files_to_join = ['train.0.json', 'train.1.json']

# Load JSON files into DataFrames
dataframes=[]
for filename in files_to_join:
    if filename.endswith('json'):
        file = pd.read_json(os.path.join(folder_path, filename), lines=True)
        dataframes.append(file)

# Concatenate DataFrames
df = pd.concat(dataframes, ignore_index=True)

df.head(5)


# In[46]:


import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('punkt')

# Load the English model for SpaCy
nlp = spacy.load("en_core_web_sm")


# In[47]:


def remove_special_characters(text):
    text = re.sub(r'http\S+|www\S+|@\S+|<.*?>', '', text) # Remove html tags and url
    text = re.sub(r'\b\w+@\w+\.\w+\b', '', text)  # Remove email addresses
    text = re.sub(r'@\w+', '', text)  # Remove usernames starting with '@'
    text = re.sub(r'\bu/\w+\b', '', text)  # Remove Reddit usernames starting with 'u/'
    text = re.sub(r'\s*\(\s*', ' ', text)  # Remove opening parenthesis and space
    text = re.sub(r'\s*\)\s*', ' ', text)   # Remove closing parenthesis and space
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with a single space
    return text

df['document'] = df['document'].apply(remove_special_characters)
df['summary'] = df['summary'].apply(remove_special_characters)


# In[48]:


import string

def remove_punctuation(text):
    # Specify which punctuation to keep
    # Keeping period, comma, exclamation mark, and question mark
    keep_punctuation = {'.', ',', '!', '?'}

    # Create a translation table for removing punctuation
    # Remove all punctuation except those in keep_punctuation
    translator = str.maketrans('', '', ''.join(set(string.punctuation) - keep_punctuation))

    # Use the translate method to remove punctuation
    return text.translate(translator)

# Apply the function to the 'document' and 'summary' columns
df['document'] = df['document'].apply(remove_punctuation)
df['summary'] = df['summary'].apply(remove_punctuation)


# In[49]:


df['document'] = df['document'].apply(lambda x: x.lower())
df['summary'] = df['summary'].apply(lambda x: x.lower())


# In[50]:


import inflect

# Initialize the inflect engine
p = inflect.engine()

# Function to convert numeric values to words
def number_to_words(text):
    """Convert numbers in the text to words."""
    # Handle abbreviations like '18f' or '18m' for age and gender
    text = re.sub(r'(\d+)(f)', lambda m: p.number_to_words(m.group(1)) + '-year-old female', text)

    # Convert 'm' to 'male' only if it appears after a number and is not part of 'meter'
    text = re.sub(r'(\d+)(m)(?![a-zA-Z])', lambda m: p.number_to_words(m.group(1)) + '-year-old male', text)

    # Convert numbers to words
    text = re.sub(r'\b\d+\b', lambda x: p.number_to_words(x.group()), text)

    return text

df['document'] = df['document'].apply(number_to_words)
df['summary'] = df['summary'].apply(number_to_words)


# In[51]:


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

# Compile all slang replacements into a regex pattern
slang_pattern = r'\b(' + '|'.join(re.escape(slang) for slang in slang_dictionary.keys()) + r')\b'
replacement_dict = {slang: full for slang, full in slang_dictionary.items()}

# Function to replace slangs using a single regex pattern
def replace_slangs(text):
    return re.sub(slang_pattern, lambda x: replacement_dict[x.group(0)], text)

# Apply the function to your dataframe
df['document'] = df['document'].apply(replace_slangs)
df['summary'] = df['summary'].apply(replace_slangs)

