from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(prompt):
    # Load the fine-tuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Encode the input and generate a response
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # Decode and print the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    prompt = "what's good?"
    print(generate_response(prompt))
