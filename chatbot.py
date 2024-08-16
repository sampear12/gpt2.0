# Author - Samika Sanghvi. All rights reserved.
# This script is provided for educational purposes and personal use. 
# For questions, feel free to reach out to me at sps76@pitt.edu :)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def chat_with_model():
    model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_gpt2')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    print("You can start chatting with Samika now. Type 'quit' to stop.")

    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        # Add the user input to the conversation history
        conversation_history.append(f"User: {user_input}\nSamika: ")

        # Limit conversation history to the last 3 exchanges to avoid context window exhaustion
        if len(conversation_history) > 3:
            conversation_history = conversation_history[-3:]

        conversation_context = "".join(conversation_history)

        input_ids = tokenizer.encode(conversation_context, return_tensors='pt')
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[-1] + 100,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            top_k=50,  # Moderate variety
            top_p=0.5,  # Encourage randomness
            temperature=0.8,  # Slightly higher temperature
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2  # Introduce a repetition penalty
        )

        # Extract the model's output, starting from where the input ends
        generated_output = output[0][input_ids.shape[-1]:]
        response = tokenizer.decode(generated_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Filter out non-ASCII characters
        response = remove_non_ascii(response).strip()

        # Check if response is empty, and if so, retry generation
        if not response:
            print("Samika didn't respond. Retrying...")
            continue

        # Add the model's response to the conversation history
        conversation_history[-1] += f"{response}\n"
        
        print(f"Samika: {response}")

if __name__ == "__main__":
    chat_with_model()
