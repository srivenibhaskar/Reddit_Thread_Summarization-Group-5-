import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
import re
import nltk
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Download required NLTK resources for text processing and cleaning
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load SpaCy model for tokenization and named entity recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Function to clean input text by removing unwanted characters, URLs, and other noise
def remove_special_characters(text):
    text = re.sub(r'http\S+|www\S+|@\S+|<.*?>', '', text)  # Remove HTML tags and URLs
    text = re.sub(r'\b\w+@\w+\.\w+\b', '', text)  # Remove email addresses
    text = re.sub(r'@\w+', '', text)  # Remove usernames starting with '@'
    text = re.sub(r'\bu/\w+\b', '', text)  # Remove Reddit usernames starting with 'u/'
    text = re.sub(r'\s*\(\s*', ' ', text)  # Remove opening parenthesis and space
    text = re.sub(r'\s*\)\s*', ' ', text)  # Remove closing parenthesis and space
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

# Define a dictionary of common slang terms and their full meanings
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
}

# Compile a regex pattern to match slang words in the input text
slang_pattern = r'\b(' + '|'.join(re.escape(slang) for slang in slang_dictionary.keys()) + r')\b'
replacement_dict = {slang: full for slang, full in slang_dictionary.items()}

# Function to replace slang words in text with their full meanings
def replace_slangs(text):
    return re.sub(slang_pattern, lambda x: replacement_dict[x.group(0)], text)

# Load the pre-trained BART model and tokenizer for text summarization
def load_model():
    model_path = r"D:\DSMM Study Material\DSMM Projects\Sem-3 Final Project\Capstone Project\Final Demo\bart_model"
    tokenizer_path = r"D:\DSMM Study Material\DSMM Projects\Sem-3 Final Project\Capstone Project\Final Demo\bart_tokenizer"

    # Load the model with updated configuration settings
    config = BartConfig.from_pretrained(model_path)
    config.update({"early_stopping": True})  # Enable early stopping for summarization
    config.update({"length_penalty": 1.0})  # Control the length of generated summaries

    model = BartForConditionalGeneration.from_pretrained(model_path, config=config)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

# Function to generate a summary of the input text using the BART model
def generate_summary(input_text, model, tokenizer):
    # Tokenize the input text while handling padding and truncation
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="longest")
    
    # Set parameters for summary generation to ensure quality and relevance
    max_len = 150  # Set maximum summary length
    min_len = 80   # Set minimum summary length
    length_penalty = 0.8  # Reduce penalty for longer summaries
    num_beams = 6  # Use beam search to explore more summarization options
    
    # Ensure the model is in evaluation mode
    model.eval()

    # Generate the summary using the specified parameters
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=max_len, 
        min_length=min_len, 
        num_beams=num_beams, 
        early_stopping=True, 
        length_penalty=length_penalty,
        no_repeat_ngram_size=2
    )

    # Decode the summary and return the resulting text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to generate hashtags from the summary by extracting relevant entities and keywords
def generate_hashtags(summary):
    # Parse the summary using SpaCy for NER and noun phrase extraction
    doc = nlp(summary)
    
    # Extract named entities (people, organizations, locations)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    
    # Extract noun phrases (contextually relevant words)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    # Filter out common stopwords and redundant words
    filtered_keywords = [word for word in entities + noun_phrases if word.lower() not in ENGLISH_STOP_WORDS]
    
    # Remove duplicates and sort the keywords
    unique_keywords = list(set(filtered_keywords))
    
    # Generate meaningful hashtags from the keywords
    hashtags = ["#" + word.replace(" ", "").lower() for word in unique_keywords[:3]]
    
    return hashtags

# Streamlit application to interact with the user
def streamlit_app():
    # Display the app title and description
    st.title("InsideThread")  # Application title
    st.subheader("Where Data Meets Dialogue")  # App tagline

    # Display a brief description of the app's functionality
    st.markdown("InsideThreads extracts the essence of Reddit conversations, transforming endless threads into clear, concise summaries.")  # Description
    
    # Instruction dropdown for user guidance
    with st.expander("Instructions 📜"):
        st.markdown("""
        1. **Enter a Reddit thread**: Paste the Reddit conversation into the text box provided.
        2. **Generate Summary**: Click on the button to generate a concise summary of the thread.
        3. **Generate Hashtags**: Click the button to generate hashtags based on the key topics in the summary.
        4. **Share**: Once you have the summary, you can share it directly on Twitter.
        """)

    # Input area for Reddit thread
    reddit_thread = st.text_area("Enter Reddit Thread", height=200, placeholder="Paste the Reddit conversation here...")

    # Display the summary if it's already generated and stored in session state
    if 'summary' in st.session_state:
        st.subheader("Generated Summary:")
        st.write(st.session_state.summary)

    # Button to generate summary from the provided Reddit thread
    if st.button("Generate Summary"):
        if reddit_thread:
            # Clean the input text by removing unwanted characters and replacing slang
            cleaned_text = remove_special_characters(reddit_thread)
            cleaned_text = replace_slangs(cleaned_text)

            # Load the pre-trained model and tokenizer
            model, tokenizer = load_model()
            
            # Generate the summary
            summary = generate_summary(cleaned_text, model, tokenizer)
            
            # Store the generated summary in session state
            st.session_state.summary = summary

            # Show the generated summary to the user
            st.subheader("Generated Summary:")
            st.write(summary)

    # Button to generate hashtags based on the summary
    if st.button("Generate Hashtags"):
        if 'summary' in st.session_state:
            # Generate hashtags based on the summary
            hashtags = generate_hashtags(st.session_state.summary)
            
            # Display the generated hashtags
            st.subheader("Generated Hashtags:")
            st.write(", ".join(hashtags))
            
    # Add a button to allow users to share the summary on Twitter
    if 'summary' in st.session_state:
        twitter_url = f"https://twitter.com/intent/tweet?text={st.session_state.summary}"
        st.markdown(f'<a href="{twitter_url}" target="_blank"><button style="background-color: #1DA1F2; color: white; font-size: 16px; padding: 10px 20px; border-radius: 5px; border: none;">Share on Twitter</button></a>', unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
