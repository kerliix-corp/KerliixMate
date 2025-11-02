# kerliixmate_local_ai.py

import json
import readline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
dataset_file = "kerliixmate_seed.jsonl"
dataset = []

with open(dataset_file, 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(json.loads(line.strip()))

# Prepare instructions for embeddings
instructions = [item['instruction'] for item in dataset]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight and fast

# Generate embeddings for all instructions
instruction_embeddings = model.encode(instructions)

def find_response(user_input, dataset, instruction_embeddings, model, top_k=1):
    """
    Find the most semantically similar instruction in the dataset and return its response.
    """
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, instruction_embeddings)[0]
    best_idx = np.argmax(similarities)
    
    if similarities[best_idx] < 0.6:
        return "I’m sorry, I didn’t quite get that. Could you rephrase or ask something else?"
    return dataset[best_idx]['response']

def chat():
    print("Welcome to Kerliixmate AI! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Kerliixmate: Goodbye! 👋")
            break
        response = find_response(user_input, dataset, instruction_embeddings, model)
        print(f"Kerliixmate: {response}")

if __name__ == "__main__":
    chat()
