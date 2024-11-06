# -*- coding: utf-8 -*-
"""model_building_and_evaluation_7E7D_layers.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nYT_8uWFekqPbYPmq7nyb6qt-8BkCDKQ
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
data= pd.read_csv(r"/content/drive/MyDrive/updated_cleaned_data.csv")
data.head()

data.info()

df = data.sample(n=10000, random_state=42)

df.info()

model_df = df.copy()

# Specify the relevant columns in the new dataset
relevant_columns = ['document', 'summary']

# Filter the DataFrame to keep only the relevant columns
model_df = model_df[relevant_columns]

# Display the first few rows of the new DataFrame
model_df.head()

#model_df = model_df.rename(columns={'lemmatized_document': 'document', 'lemmatized_summary': 'summary'})

# displaying the first few rows of the updated DataFrame
#model_df.head()

texts = model_df['document'].dropna().tolist()

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader

# loading the pretrained BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# loading the pre-trained BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# freezing the first 7 layers of Encoder and Decoder for Transfer Learning
for layer in model.model.encoder.layers[:7]:
    for param in layer.parameters():
        param.requires_grad = False

for layer in model.model.decoder.layers[:7]:
    for param in layer.parameters():
        param.requires_grad = False

# customizing the model for summarization on Reddit dataset
model.config.max_length = 100                                          # max length of generated summaries
model.config.min_length = 20                                           # min length of generated summaries
model.config.no_repeat_ngram_size = 3                                  # to avoid repeating trigrams
model.config.early_stopping = True

# splitting the data into training and validation datasets
from sklearn.model_selection import train_test_split
train_data, eval_data = train_test_split(model_df, test_size=0.2)

import torch
from torch.utils.data import Dataset, DataLoader

class RedditSummaryDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)  # Reset index to avoid key errors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    # Access the document and summary from the DataFrame
        document = str(self.data.loc[idx, 'document'])
        summary = str(self.data.loc[idx, 'summary'])

    # Tokenize document and summary using the correct tokenizer
        inputs = self.tokenizer(
        document,
        max_length=self.max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
        )
        labels = self.tokenizer(
        summary,
        max_length=self.max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
        ).input_ids

    # Make sure to squeeze the inputs to remove unnecessary dimensions
        inputs['input_ids'] = inputs['input_ids'].squeeze()
        inputs['attention_mask'] = inputs['attention_mask'].squeeze()
        inputs['labels'] = labels.squeeze()  # Ensure labels have the correct shape
        return inputs

# creating datasets for training and evaluation
train_dataset = RedditSummaryDataset(train_data, tokenizer)
eval_dataset = RedditSummaryDataset(eval_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Save the model
model.save_pretrained('/content/drive/MyDrive/bart_7E7D_model')

# Save the tokenizer
tokenizer.save_pretrained('/content/drive/MyDrive/bart_7E7D_tokenizer')

import warnings
import evaluate
from nltk.translate.bleu_score import sentence_bleu
import torch
warnings.filterwarnings("ignore", category=UserWarning)
# Load ROUGE metric
rouge = evaluate.load("rouge")

# Define a function to calculate BLEU score
def calculate_bleu(reference_summary, generated_summary):
    # Tokenize the reference and generated summaries
    reference_tokens = reference_summary.split()  # Assuming reference_summary is a string
    generated_tokens = generated_summary.split()  # Assuming generated_summary is a string

    # Calculate BLEU score
    return sentence_bleu([reference_tokens], generated_tokens)

# Function to evaluate the model on the eval_dataset
def evaluate_model(model, tokenizer, test_loader):
    # Setting the model to evaluation mode
    model.eval()
    generated_summaries = []
    reference_summaries = []
    bleu_scores = []

    # Loop through the test_loader for batch processing
    for batch in test_loader:
        # Move inputs to the device (CPU or GPU)
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        reference_summaries_batch = batch['labels']

        # Generate summaries without gradient calculation
        with torch.no_grad():
            summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                         max_length=100, min_length=20, length_penalty=2.0,
                                         num_beams=4, early_stopping=True)

        # Decode generated summaries and append to the list
        for idx in range(len(summary_ids)):
            generated_summary = tokenizer.decode(summary_ids[idx], skip_special_tokens=True)
            generated_summaries.append(generated_summary)

            # Decode the reference summary and calculate BLEU score
            reference_summary = tokenizer.decode(reference_summaries_batch[idx], skip_special_tokens=True)
            reference_summaries.append(reference_summary)

            bleu_score = calculate_bleu(reference_summary, generated_summary)
            bleu_scores.append(bleu_score)

    # Calculate average BLEU score
    average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    # Calculate the ROUGE scores
    rouge_results = rouge.compute(predictions=generated_summaries, references=reference_summaries)
    return rouge_results, average_bleu_score

# Assuming eval_loader is a DataLoader created from the RedditSummaryDataset
rouge_scores, average_bleu = evaluate_model(model, tokenizer, eval_loader)
print("ROUGE Scores:", rouge_scores)
print("Average BLEU Score:", average_bleu)

"""After freezing 7 layers of the BART model, the training and validation loss values over 3 epochs are as follows:
- **Epoch 1**: Training Loss = 0.3048, Validation Loss = 0.3521
- **Epoch 2**: Training Loss = 0.3115, Validation Loss = 0.3406
- **Epoch 3**: Training Loss = 0.2794, Validation Loss = 0.3441

With 7 layers frozen, the model's training loss shows a slight decrease overall, indicating continued learning with reduced model complexity. The validation loss initially decreases but then increases slightly in the third epoch, which may hint at early signs of overfitting. Freezing layers appears to help manage the model’s capacity, though further training might be needed to observe the trend more clearly.

The updated ROUGE and BLEU scores after freezing 7 layers in the BART model indicate the following:
- **ROUGE-1**: 0.2495, showing about 24.9% match in individual words.
- **ROUGE-2**: 0.0698, indicating about 7% match in two-word phrases.
- **ROUGE-L** and **ROUGE-Lsum**: 0.1815 and 0.1812, reflecting around 18.1% similarity in sentence structure.

The **BLEU score** is 0.0156, which remains low and suggests limited overlap in word choice and phrasing between the generated and reference text. Freezing 7 layers may have helped manage model complexity, but it has not significantly boosted lexical alignment with the reference.
"""