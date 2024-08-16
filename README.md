# GPT-2 Chatbot

This guide provides step-by-step instructions for setting up a local GPT-2 model, fine-tuning it with custom data, and deploying it as a chatbot that runs in the command prompt.

## Prerequisites

Ensure you have the following installed:

- **Python 3.8 or higher**
- **Git** (for cloning the repository)
- **Conda** (recommended for managing environments)

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Create and Activate a Virtual Environment**
Using conda:
```bash
conda create -n gpt2_chatbot python=3.8
conda activate gpt2_chatbot
```
Or using venv:
```bash
python -m venv gpt2_chatbot_env
source gpt2_chatbot_env/bin/activate  # On Windows use: gpt2_chatbot_env\Scripts\activate
```

3. **Install Required Libraries** 
Install the necessary Python libraries using pip:
```bash
pip install torch transformers datasets
```

4. **Prepare Your Dataset**
Prepare Your Dataset
Prepare your dataset in JSON format where each entry contains an input and an output. For example:
```json
{
    "input": "Hello, how are you?",
    "output": "I am fine, thank you!"
}
```
Save this file as data.json in the root directory of the repository or modify the one already present.

5. **Fine-Tune the model**
Run the fine_tune.py script to fine-tune the GPT-2 model on your custom dataset:

```bash
python fine_tune.py
```

This script will:
-Load and tokenize your dataset.
-Fine-tune the GPT-2 model.
-Save the fine-tuned model and tokenizer to the ./fine_tuned_gpt2 directory.

5. **Deploy the chatbot**
After fine-tuning, deploy the chatbot using the chatbot.py script:
```bash
python chatbot.py
```
This script will:
-Load the fine-tuned model and tokenizer.
-Start a command-line interface for chatting with the model.

Then you can start chatting with the model!

**Notes**
-Ensure your GPU drivers are up to date if you plan to use GPU acceleration.
-Modify the fine_tune.py script's training parameters as needed for your dataset.

Feel free to reach out to me on sps76@pitt.edu for any questions regarding the scripts :)
