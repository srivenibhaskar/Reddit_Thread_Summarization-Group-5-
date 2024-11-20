#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install transformers datasets torch


# In[8]:


import pandas as pd
df= pd.read_csv('/content/final_data.csv', on_bad_lines='skip')
df.head()


# In[ ]:


df.columns


# In[ ]:


model_df = model_df = df.copy()
relevant_columns = ['Lemmatized_document', 'lemmatized_summary_y']  # Modify this list based on your needs

# Filter the DataFrame to keep only the relevant columns
model_df = model_df[relevant_columns]

# Display the first few rows of the new DataFrame
model_df.head()


# In[ ]:


model_df = model_df.rename(columns={'Lemmatized_document': 'document', 'lemmatized_summary_y': 'summary'})

# Display the first few rows of the updated DataFrame
model_df.head()


# In[ ]:


from transformers import BartTokenizer

# Load the pre-trained BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


# In[ ]:


from sklearn.model_selection import train_test_split

# Split the data
train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)


# In[ ]:


# Tokenize the Dataset
def tokenize_function(examples):
    # Tokenize input documents (using the 'document' column)
    inputs = tokenizer(examples['document'], max_length=1024, truncation=True, padding="max_length")

    # Tokenize target summaries (using the 'summary' column)
    labels = tokenizer(examples['summary'], max_length=150, truncation=True, padding="max_length")

    # Replace pad tokens with -100 in labels to ignore them during loss calculation
    labels["input_ids"] = [
        (label if label != tokenizer.pad_token_id else -100) for label in labels["input_ids"]
    ]

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"]
    }



# In[ ]:


# Clean the 'document' and 'summary' columns
train_df['document'] = train_df['document'].fillna('').astype(str)
train_df['summary'] = train_df['summary'].fillna('').astype(str)

test_df['document'] = test_df['document'].fillna('').astype(str)
test_df['summary'] = test_df['summary'].fillna('').astype(str)


# In[ ]:


# Apply Tokenization to Training and Validation Sets
from datasets import Dataset

# Convert Pandas DataFrames to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Apply the tokenization function to the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)


# In[ ]:


# BART Model Configuration
from transformers import BartConfig, BartForConditionalGeneration

# Define BART configuration for training from scratch
config = BartConfig(
    vocab_size=tokenizer.vocab_size,  # The tokenizer's vocabulary size
    max_position_embeddings=1024,     # Maximum input length
    encoder_layers=6,                 # Number of encoder layers
    decoder_layers=6,                 # Number of decoder layers
    d_model=768,                      # Hidden dimension size
    pad_token_id=tokenizer.pad_token_id,  # Padding token from tokenizer
    eos_token_id=tokenizer.eos_token_id,  # End of sentence token from tokenizer
)

# Initialize the BART model with the configuration
model = BartForConditionalGeneration(config)


# In[ ]:


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',  # Evaluate every few steps
    learning_rate=5e-5,
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=3,              # Start with 3 epochs
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1,              # Save only the last model
    eval_steps=500,                  # Evaluate every 500 steps
    save_steps=500,                  # Save every 500 steps
    fp16=True,                       # Enable mixed precision training
    gradient_accumulation_steps=2,   # Use gradient accumulation
)


# In[ ]:


from transformers import Trainer

# Initialize the Trainer with the model, training arguments, and datasets
trainer = Trainer(
    model=model,                     # BART model initialized from scratch
    args=training_args,              # Training arguments
    train_dataset=train_dataset,     # Training dataset
    eval_dataset=test_dataset,       # Evaluation dataset (use eval_dataset for validation during training)
    tokenizer=tokenizer,             # The tokenizer to use
)

# Train the model
trainer.train()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




