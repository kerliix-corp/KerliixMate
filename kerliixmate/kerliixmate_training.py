# kerliixmate_training.py

import json
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Load .jsonl dataset
dataset_file = "kerliixmate_seed.jsonl"
examples = []

with open(dataset_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())
        examples.append({"instruction": item["instruction"], "response": item["response"]})

# Split into train and validation sets (80/20)
train_data, val_data = train_test_split(examples, test_size=0.2, random_state=42)

# Create Hugging Face Dataset objects
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")





from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"  # Lightweight and fast
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)





max_length = 128

def preprocess(example):
    inputs = example['instruction']
    targets = example['response']
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=max_length, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = train_dataset.map(preprocess, remove_columns=["instruction", "response"])
val_dataset = val_dataset.map(preprocess, remove_columns=["instruction", "response"])



from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./kerliixmate_model",
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    report_to=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()


model.save_pretrained("./kerliixmate_model")
tokenizer.save_pretrained("./kerliixmate_model")



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("./kerliixmate_model")
tokenizer = AutoTokenizer.from_pretrained("./kerliixmate_model")

input_text = "How do I reset my password?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uuid

# ---------- Load Fine-Tuned Model ----------
model_path = "./kerliixmate_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# ---------- FastAPI App ----------
app = FastAPI(title="Kerliixmate AI with Context", version="3.0")

# Store conversation history in memory (session_id -> messages)
conversations = {}
MAX_HISTORY = 5  # Keep last 5 exchanges

class UserQuery(BaseModel):
    session_id: str = None  # optional
    message: str

def generate_response(user_input, history, max_length=128):
    # Combine conversation history with current input
    conversation_text = ""
    for turn in history:
        conversation_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    conversation_text += f"User: {user_input}\nAssistant:"

    inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    # Create new session if none provided
    session_id = query.session_id or str(uuid.uuid4())

    if session_id not in conversations:
        conversations[session_id] = []

    # Generate response
    history = conversations[session_id]
    response = generate_response(query.message, history)

    # Update conversation history
    conversations[session_id].append({"user": query.message, "assistant": response})
    if len(conversations[session_id]) > MAX_HISTORY:
        conversations[session_id] = conversations[session_id][-MAX_HISTORY:]

    return {"session_id": session_id, "response": response}

@app.get("/")
async def root():
    return {"message": "Kerliixmate AI with multi-turn context. Use POST /chat with session_id & message."}





import json
from datetime import datetime

LOG_FILE = "kerliixmate_logs.jsonl"

def log_interaction(session_id, user_message, assistant_response):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "user_message": user_message,
        "assistant_response": assistant_response
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
