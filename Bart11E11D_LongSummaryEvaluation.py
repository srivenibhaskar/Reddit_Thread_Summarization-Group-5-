#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


df = pd.read_csv('/content/drive/MyDrive/updated_cleaned_data.csv')
df.head(5)


# In[ ]:


from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
import torch

# Set a fixed random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# Step 2: Select a smaller subset of the dataset
subset_size = 12000  # Choose the size of your subset
subset_df = df.sample(n=subset_size, random_state=random_seed)  # Randomly select records

# Step 3: Split the dataset into training and testing sets
train_df, test_df = train_test_split(subset_df, test_size=0.2, random_state=random_seed)


# In[ ]:


# Step 4: Load the BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Function to tokenize the text data
def tokenize_function(examples):
    documents = examples['document'].astype(str).tolist()
    summaries = examples['summary'].astype(str).tolist()

    # Tokenize the documents and summaries
    model_inputs = tokenizer(documents, max_length=1024, truncation=True, padding=True)
    labels = tokenizer(summaries, max_length=150, truncation=True, padding=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize the training and test data
train_encodings = tokenize_function(train_df)
test_encodings = tokenize_function(test_df)


# In[ ]:


# Step 5: Prepare the data for the Trainer
class SummaryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = SummaryDataset(train_encodings)
test_dataset = SummaryDataset(test_encodings)


# In[ ]:


# Step 6: Load the BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Step 7: Freeze layers
# Freeze the first 11 layers of the encoder
for layer in model.model.encoder.layers[:11]:  # Access encoder through model.model
    for param in layer.parameters():
        param.requires_grad = False

# Freeze the first 9 layers of the decoder
for layer in model.model.decoder.layers[:11]:  # Access decoder through model.model
    for param in layer.parameters():
        param.requires_grad = False

# Step 8: Define training arguments with mixed precision
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,             # Enable mixed precision training
)

# Step 9: Create Trainer instance

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Step 10: Train the model
trainer.train()

# Optional: Evaluate the model (if needed)
results = trainer.evaluate()
print(results)


# In[ ]:


# Save the model
model.save_pretrained('/content/drive/MyDrive/Project/model_files/bart_11E11D_model')

# Save the tokenizer
tokenizer.save_pretrained('/content/drive/MyDrive/Project/model_files/bart_11E11D_tokenizer')


# In[ ]:


get_ipython().system('pip install rouge-score')


# In[ ]:


import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


# In[ ]:


from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import nltk
from sklearn.metrics import f1_score
from transformers import BertTokenizer

# Download necessary NLTK data files
nltk.download('punkt')

# Initialize ROUGE scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Define device and move model to that device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to generate summaries
def generate_summary(text, max_length=150):
    model.eval()  # Set model to evaluation mode

    # Tokenize the input and move to the correct device
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(device)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=4, length_penalty=2.0)

    # Decode the generated summary and return
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Calculate ROUGE, BLEU, and BERT F1 scores
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
bleu_scores = []
bert_f1_scores = []

for idx, row in test_df.iterrows():
    # Generate model summary
    generated_summary = generate_summary(row['document'])

    # Reference summary
    reference_summary = row['summary']

    # Calculate ROUGE scores
    scores = rouge_scorer.score(reference_summary, generated_summary)
    rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
    rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
    rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

    # Calculate BLEU score
    reference_tokens = nltk.word_tokenize(reference_summary)
    generated_tokens = nltk.word_tokenize(generated_summary)
    bleu_score = sentence_bleu([reference_tokens], generated_tokens)
    bleu_scores.append(bleu_score)

    # Tokenize using BERT tokenizer
    reference_tokens_bert = bert_tokenizer.tokenize(reference_summary)
    generated_tokens_bert = bert_tokenizer.tokenize(generated_summary)

    # Calculate BERT F1 score
    # First calculate precision and recall based on token matches
    common_tokens = set(reference_tokens_bert) & set(generated_tokens_bert)
    precision = len(common_tokens) / len(generated_tokens_bert) if len(generated_tokens_bert) > 0 else 0
    recall = len(common_tokens) / len(reference_tokens_bert) if len(reference_tokens_bert) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    bert_f1_scores.append(f1)

# Calculate average scores
avg_rouge1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
avg_rouge2 = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
avg_rougeL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_bert_f1 = sum(bert_f1_scores) / len(bert_f1_scores)

# Print evaluation results
print(f"Average ROUGE-1 Score: {avg_rouge1:.4f}")
print(f"Average ROUGE-2 Score: {avg_rouge2:.4f}")
print(f"Average ROUGE-L Score: {avg_rougeL:.4f}")
print(f"Average BLEU Score: {avg_bleu:.4f}")
print(f"Average BERT F1 Score: {avg_bert_f1:.4f}")


# In[ ]:


from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import nltk
from sklearn.metrics import f1_score
from transformers import BertTokenizer

# Download necessary NLTK data files
nltk.download('punkt')

# Initialize ROUGE scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Define device and move model to that device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to generate shorter summaries (max_length=80)
def generate_shorter_summary(text, max_length=80):
    model.eval()  # Set model to evaluation mode

    # Tokenize the input and move to the correct device
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(device)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=4, length_penalty=2.0)

    # Decode the generated summary and return
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Calculate ROUGE, BLEU, and BERT F1 scores for shorter summaries
rouge_scores_shorter = {'rouge1': [], 'rouge2': [], 'rougeL': []}
bleu_scores_shorter = []
bert_f1_scores_shorter = []

for idx, row in test_df.iterrows():
    # Generate shorter model summary
    generated_summary_shorter = generate_shorter_summary(row['document'])

    # Reference summary
    reference_summary = row['summary']

    # Calculate ROUGE scores
    scores = rouge_scorer.score(reference_summary, generated_summary_shorter)
    rouge_scores_shorter['rouge1'].append(scores['rouge1'].fmeasure)
    rouge_scores_shorter['rouge2'].append(scores['rouge2'].fmeasure)
    rouge_scores_shorter['rougeL'].append(scores['rougeL'].fmeasure)

    # Calculate BLEU score
    reference_tokens = nltk.word_tokenize(reference_summary)
    generated_tokens = nltk.word_tokenize(generated_summary_shorter)
    bleu_score = sentence_bleu([reference_tokens], generated_tokens)
    bleu_scores_shorter.append(bleu_score)

    # Tokenize using BERT tokenizer
    reference_tokens_bert = bert_tokenizer.tokenize(reference_summary)
    generated_tokens_bert = bert_tokenizer.tokenize(generated_summary_shorter)

    # Calculate BERT F1 score
    common_tokens = set(reference_tokens_bert) & set(generated_tokens_bert)
    precision = len(common_tokens) / len(generated_tokens_bert) if len(generated_tokens_bert) > 0 else 0
    recall = len(common_tokens) / len(reference_tokens_bert) if len(reference_tokens_bert) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    bert_f1_scores_shorter.append(f1)

# Calculate average scores for shorter summaries
avg_rouge1_shorter = sum(rouge_scores_shorter['rouge1']) / len(rouge_scores_shorter['rouge1'])
avg_rouge2_shorter = sum(rouge_scores_shorter['rouge2']) / len(rouge_scores_shorter['rouge2'])
avg_rougeL_shorter = sum(rouge_scores_shorter['rougeL']) / len(rouge_scores_shorter['rougeL'])
avg_bleu_shorter = sum(bleu_scores_shorter) / len(bleu_scores_shorter)
avg_bert_f1_shorter = sum(bert_f1_scores_shorter) / len(bert_f1_scores_shorter)

# Print evaluation results for shorter summaries
print(f"Average ROUGE-1 Score (shorter): {avg_rouge1_shorter:.4f}")
print(f"Average ROUGE-2 Score (shorter): {avg_rouge2_shorter:.4f}")
print(f"Average ROUGE-L Score (shorter): {avg_rougeL_shorter:.4f}")
print(f"Average BLEU Score (shorter): {avg_bleu_shorter:.4f}")
print(f"Average BERT F1 Score (shorter): {avg_bert_f1_shorter:.4f}")

