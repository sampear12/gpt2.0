# Author - Samika Sanghvi. All rights reserved.
# This script is provided for educational purposes and personal use. 
# For questions, feel free to reach out to me at sps76@pitt.edu :)

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from torch.utils.data import random_split

def load_custom_dataset(file_path, tokenizer, block_size=128):
    # Load the dataset from a JSON file
    dataset = load_dataset('json', data_files={'train': file_path}, split='train')

    # Tokenization function to process the text data
    def tokenize_function(examples):
        combined_text = [f"### Input: {inp} ### Output: {out}" for inp, out in zip(examples['input'], examples['output'])]
        tokenized_inputs = tokenizer(
            combined_text,  # Tokenize the combined input-output text
            padding="max_length",  # Pad sequences to the maximum length
            truncation=True,  # Truncate sequences longer than block_size
            max_length=block_size,  # Maximum length of sequences
            return_special_tokens_mask=True  # Return mask for special tokens
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # Use input_ids as labels
        return tokenized_inputs

    # Apply tokenization to the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["input", "output"])
    return tokenized_datasets

def main():
    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Add padding token 
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    # Load the GPT-2 model and resize token embeddings to include the padding token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    # Load and tokenize the dataset
    dataset = load_custom_dataset('data.json', tokenizer)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Define training arguments
    training_args = TrainingArguments(
        # You can change the parameters to mess around with training the model. It's a great way to understand the impact each argument has!
        output_dir='./fine_tuned_gpt2',  # Output directory for model checkpoints
        overwrite_output_dir=True,  # Overwrite the output directory
        num_train_epochs=4,  # Number of training epochs
        per_device_train_batch_size=2,  # Batch size for training
        per_device_eval_batch_size=2,  # Batch size for evaluation
        warmup_steps=200,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Weight decay for optimizer
        logging_dir='./logs',  # Directory for storing logs
        logging_steps=100,  # Log every 100 steps
        evaluation_strategy="steps",  # Evaluate the model every `eval_steps`
        eval_steps=200,  # Evaluation step interval
        save_steps=400,  # Save checkpoint every 400 steps
        save_total_limit=3,  # Limit the total number of checkpoints
        learning_rate=5e-5,  # Learning rate
        load_best_model_at_end=True,  # Load the best model at the end of training
        gradient_accumulation_steps=1,  # Number of gradient accumulation steps
        fp16=True,  # Use mixed precision training
        logging_first_step=True,  # Log the first step
    )

    # Initialize the Trainer with model, training arguments, and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Early stopping to prevent overfitting
    )

    # Train the model
    trainer.train()
    
    trainer.save_model('./fine_tuned_gpt2')
    tokenizer.save_pretrained('./fine_tuned_gpt2')

if __name__ == "__main__":
    main()
