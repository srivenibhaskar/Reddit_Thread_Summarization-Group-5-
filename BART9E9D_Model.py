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


df = pd.read_csv('/content/drive/MyDrive/Project/csv_files/updated_text_analysis.csv')
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
subset_size = 5000  # Choose the size of your subset
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
# Freeze the first 9 layers of the encoder
for layer in model.model.encoder.layers[:9]:  # Access encoder through model.model
    for param in layer.parameters():
        param.requires_grad = False

# Freeze the first 9 layers of the decoder
for layer in model.model.decoder.layers[:9]:  # Access decoder through model.model
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
    fp16=True,  # Enable mixed precision training
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
model.save_pretrained('/content/drive/MyDrive/Project/model_files/bart_9E9D_model')

# Save the tokenizer
tokenizer.save_pretrained('/content/drive/MyDrive/Project/model_files/bart_9E9D_tokenizer')


# In[ ]:




