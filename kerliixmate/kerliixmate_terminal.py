# kerliixmate_terminal.py

import json
import readline  # optional, enables arrow key navigation in terminal
from difflib import get_close_matches

# Load dataset
dataset_file = "kerliixmate_seed.jsonl"
dataset = []

with open(dataset_file, 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(json.loads(line.strip()))

def find_response(user_input, dataset, n=1, cutoff=0.6):
    """
    Find the closest instruction match in the dataset using difflib.
    Returns the corresponding response.
    """
    instructions = [item['instruction'] for item in dataset]
    matches = get_close_matches(user_input, instructions, n=n, cutoff=cutoff)
    if matches:
        for item in dataset:
            if item['instruction'] == matches[0]:
                return item['response']
    return "I’m sorry, I didn’t quite get that. Could you rephrase or ask something else?"

def chat():
    print("Welcome to Kerliixmate! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Kerliixmate: Goodbye! 👋")
            break
        response = find_response(user_input, dataset)
        print(f"Kerliixmate: {response}")

if __name__ == "__main__":
    chat()
