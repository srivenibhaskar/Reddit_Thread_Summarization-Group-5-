import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
import re
import nltk
import spacy
from rake_nltk import Rake

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load SpaCy model for tokenization
nlp = spacy.load("en_core_web_sm")

# Define the data cleaning functions
def remove_special_characters(text):
    text = re.sub(r'http\S+|www\S+|@\S+|<.*?>', '', text)  # Remove html tags and urls
    text = re.sub(r'\b\w+@\w+\.\w+\b', '', text)  # Remove email addresses
    text = re.sub(r'@\w+', '', text)  # Remove usernames starting with '@'
    text = re.sub(r'\bu/\w+\b', '', text)  # Remove Reddit usernames starting with 'u/'
    text = re.sub(r'\s*\(\s*', ' ', text)  # Remove opening parenthesis and space
    text = re.sub(r'\s*\)\s*', ' ', text)  # Remove closing parenthesis and space
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

# Define slang replacement dictionary
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

# Function to replace slangs
def replace_slangs(text):
    return re.sub(slang_pattern, lambda x: replacement_dict[x.group(0)], text)

# Load the model and tokenizer
def load_model():
    # Load model and tokenizer
    model_path = r'D:\DSMM Study Material\DSMM Projects\Sem-3 Final Project\Capstone Project\bart_9E9D_model'
    tokenizer_path = r'D:\DSMM Study Material\DSMM Projects\Sem-3 Final Project\Capstone Project\bart_9E9D_tokenizer'

    # Manually set `early_stopping` in the config before loading the model
    config = BartConfig.from_pretrained(model_path)
    config.update({"early_stopping": True})  # Set early_stopping to True
    config.update({"length_penalty": 1.0})  # Explicitly set the length_penalty (e.g., 1.0)

    model = BartForConditionalGeneration.from_pretrained(model_path, config=config)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

# Step 2: Define the function to generate summary from the input text
def generate_summary(input_text, model, tokenizer):
    """
    Function to generate a summary from a given input text using a trained model.

    Args:
    - input_text: str, the input text to summarize.
    - model: the pretrained model.
    - tokenizer: the tokenizer for the model.

    Returns:
    - summary: str, the generated summary.
    """
    # Tokenize the input text with attention masks
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="longest")
    
    # Set max_length for summary generation
    max_len = 150  # Fixed length for the summary
    
    # Ensure the model is in evaluation mode
    model.eval()

    # Generate summary IDs from the model
    summary_ids = model.generate(inputs['input_ids'], max_length=max_len, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)

    # Decode the summary and return it
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to generate hashtags from the summary using RAKE
def generate_hashtags(summary):
    """
    Function to generate hashtags from the summary using RAKE.
    """
    rake = Rake()
    rake.extract_keywords_from_text(summary)
    keywords = rake.get_ranked_phrases()  # Get the ranked key phrases

    # Generate hashtags from the key phrases
    hashtags = ['#' + phrase.replace(" ", "") for phrase in keywords[:10]]  # Limit to top 10 phrases
    return hashtags

# Streamlit UI
def streamlit_app():
    st.title("Reddit Thread Summarizer")

    # Input for Reddit thread
    reddit_thread = st.text_area("Enter Reddit Thread", height=200)

    # Button to generate summary
    if st.button("Generate Summary"):
        if reddit_thread:
            # Load the model and tokenizer
            model, tokenizer = load_model()
            
            # Generate summary
            summary = generate_summary(reddit_thread, model, tokenizer)
            
            # Store the summary in session state
            st.session_state.summary = summary

            # Show the generated summary
            st.subheader("Generated Summary:")
            st.write(summary)

    # Check if summary exists in session state
    if 'summary' in st.session_state:
        # Option to generate hashtags
        if st.button("Generate Hashtags"):
            hashtags = generate_hashtags(st.session_state.summary)
            st.subheader("Generated Hashtags:")
            st.write(hashtags)  # Display hashtags as a list
    
            # Twitter Button
            hashtags_text = " ".join(hashtags)
            twitter_url = f"https://twitter.com/intent/tweet?text={st.session_state.summary}%20{hashtags_text}"
            st.markdown(f"<a href='{twitter_url}' target='_blank'><button class='twitter-btn'>Share on Twitter</button></a>", unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
