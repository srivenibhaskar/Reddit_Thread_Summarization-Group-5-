#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pandas as pd
df= pd.read_csv(r"/content/drive/MyDrive/updated_cleaned_data.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


model_df = df.copy()

# Specify the relevant columns in the new dataset
relevant_columns = ['document', 'summary']

# Filter the DataFrame to keep only the relevant columns
model_df = model_df[relevant_columns]

# Display the first few rows of the new DataFrame
model_df.head()


# In[ ]:


import torch

# Check if GPU is available and set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


# In[ ]:


from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
import torch
import pandas as pd

# Step 2: Select a smaller subset of the dataset
subset_size = 10000  # Choose the size of your subset
subset_df = df.sample(n=subset_size, random_state=42)  # Randomly select records
#subset_df = df

# Step 3: Split the dataset into training and testing sets
train_df, test_df = train_test_split(subset_df, test_size=0.2, random_state=42)

# Step 4: Load the BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Function to tokenize the text data
def tokenize_function(examples,summary_max_length=150):
    documents = examples['document'].astype(str).tolist()
    summaries = examples['summary'].astype(str).tolist()

    # Tokenize the documents and summaries
    model_inputs = tokenizer(documents, max_length=1024, truncation=True, padding=True)
    labels = tokenizer(summaries, max_length=100, truncation=True, padding=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize the training and test data
train_encodings = tokenize_function(train_df,summary_max_length=150)
test_encodings = tokenize_function(test_df,summary_max_length=150)


# In[ ]:


from transformers import AdamW

# Step 4: Prepare the data for the Trainer
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


from transformers import BartForConditionalGeneration

# Load pre-trained BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# Freeze the first 10 encoder layers
for i in range(10):
    for param in model.model.encoder.layers[i].parameters():
        param.requires_grad = False


# Freeze the first 10 decoder layers
for i in range(10):
    for param in model.model.decoder.layers[i].parameters():
        param.requires_grad = False

# Check if the layers are frozen
for i in range(10):
    print(f"Encoder layer {i+1} frozen: {[p.requires_grad for p in model.model.encoder.layers[i].parameters()]}")
    print(f"Decoder layer {i+1} frozen: {[p.requires_grad for p in model.model.decoder.layers[i].parameters()]}")


# In[ ]:


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,                      # Enabled mixed precision
)

# Step 8: Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


# In[ ]:


# Step 9: Train the model
trainer.train()


# In[ ]:


# Save the model
model.save_pretrained('/content/drive/MyDrive/bart_10E10D(10)_model')

# Save the tokenizer
tokenizer.save_pretrained('/content/drive/MyDrive/bart_10E10D(10)_tokenizer')


# In[ ]:


import nltk
nltk.download('punkt')
nltk.download('punkt_tab')  # Add this if it prompts for 'punkt_tab'


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

