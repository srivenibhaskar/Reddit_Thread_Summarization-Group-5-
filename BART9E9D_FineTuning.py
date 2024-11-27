#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np


# In[ ]:


df = pd.read_csv('/content/drive/MyDrive/Project/csv_files/updated_text_analysis.csv')
df.head(5)


# In[ ]:


from sklearn.model_selection import train_test_split
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
import torch

# Load the saved model and tokenizer
model_path = "/content/drive/MyDrive/Project/model_files/bart_9E9D_model"  # Replace with your saved model path
tokenizer_path = "/content/drive/MyDrive/Project/model_files/bart_9E9D_tokenizer"  # Replace with your saved tokenizer path

model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(tokenizer_path)

# Define the device and move the model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set a fixed random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# Step 2: Select a smaller subset of the dataset
subset_size = 5000  # Choose the size of your subset
subset_df = df.sample(n=subset_size, random_state=random_seed)  # Randomly select records

# Step 3: Split the dataset into training and testing sets
train_df, test_df = train_test_split(subset_df, test_size=0.2, random_state=random_seed)


# In[ ]:


# Tokenize function (similar to before)
def tokenize_function(examples):
    documents = examples['document'].astype(str).tolist()
    summaries = examples['summary'].astype(str).tolist()

    model_inputs = tokenizer(documents, max_length=1024, truncation=True, padding=True)
    labels = tokenizer(summaries, max_length=150, truncation=True, padding=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize the training and test data
train_encodings = tokenize_function(train_df)
test_encodings = tokenize_function(test_df)


# In[ ]:


# Dataset class
class SummaryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Create train and test datasets
train_dataset = SummaryDataset(train_encodings)
test_dataset = SummaryDataset(test_encodings)


# In[ ]:


# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Define a very low learning rate
learning_rate = 1e-6  # Very low learning rate for fine-tuning

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',           # Output directory
    evaluation_strategy="epoch",      # Evaluate after each epoch
    learning_rate=learning_rate,     # Set the learning rate to the low value
    per_device_train_batch_size=4,    # Batch size for training
    per_device_eval_batch_size=4,     # Batch size for evaluation
    num_train_epochs=3,               # Number of epochs
    weight_decay=0.01,                # Weight decay for regularization
    fp16=True,                        # Enable mixed precision training (if supported)
    save_steps=500,                   # Save checkpoint every 500 steps
    logging_dir='./logs',             # Directory for storing logs
)

# Create the Trainer instance
trainer = Trainer(
    model=model,                      # The model to train
    args=training_args,               # Training arguments
    train_dataset=train_dataset,      # Training dataset (already tokenized)
    eval_dataset=test_dataset         # Evaluation dataset (already tokenized)
)

# Fine-tune the model
trainer.train()

# Optionally, you can evaluate after fine-tuning
results = trainer.evaluate()
print(results)


# In[ ]:




