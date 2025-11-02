import json
import os

DATASET_FILE = "../datasets/kerliixmate_seed.jsonl"

def list_examples():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            item = json.loads(line.strip())
            print(f"{idx}. Instruction: {item['instruction']}, Response: {item['response']}")

def add_example(instruction, response, category="general"):
    new_entry = {"instruction": instruction, "response": response, "category": category}
    with open(DATASET_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(new_entry) + "\n")
    print("Example added successfully!")

def delete_example(index):
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if 0 <= index-1 < len(lines):
        removed = lines.pop(index-1)
        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Deleted: {removed}")
    else:
        print("Invalid index")

# Example usage
if __name__ == "__main__":
    # List all
    # list_examples()

    # Add example
    # add_example("How do I cancel my subscription?", "You can cancel in Billing → Manage Subscription.")

    # Delete example
    # delete_example(5)
    pass
