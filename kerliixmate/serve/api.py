from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- Load Fine-Tuned Model ----------
model_path = "./kerliixmate_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# ---------- FastAPI App ----------
app = FastAPI(title="Kerliixmate AI API", version="2.0")

class UserQuery(BaseModel):
    message: str

def generate_response(user_input, max_length=128):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128)
    outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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

    # Log interaction
    log_interaction(session_id, query.message, response)

    return {"session_id": session_id, "response": response}

@app.get("/")
async def root():
    return {"message": "Welcome to Kerliixmate AI API v2! Use POST /chat with {'message': 'your text'}"}




from collections import Counter
import json

queries = []
with open("kerliixmate_logs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        log = json.loads(line)
        queries.append(log["user_message"])

top_queries = Counter(queries).most_common(10)
print(top_queries)
