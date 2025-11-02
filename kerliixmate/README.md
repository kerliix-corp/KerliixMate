# Kerliixmate – AI Assistant for Kerliix Corporation

Kerliixmate is a fully local AI assistant designed for Kerliix Corporation. It can answer customer questions, handle greetings, support payments flow, escalate issues, and more. It supports **multi-turn conversations**, session persistence, logging, and fine-tuning with your own datasets.

---

## Features

* Pre-trained on a **seed dataset** with FAQs, greetings, payments, and escalation phrases.
* **Fine-tunable locally** on `.jsonl` datasets.
* **FastAPI backend API** for integration with web or mobile frontends.
* **Persistent session context** for multi-turn conversations.
* **Logging and analytics** of all interactions.
* **Admin tools** to manage datasets, retrain models, and monitor sessions.

---

## Table of Contents

* [Installation](#installation)
* [Dataset](#dataset)
* [Training / Fine-Tuning](#training--fine-tuning)
* [Running the API](#running-the-api)
* [Admin Tools](#admin-tools)
* [Terminal Chat Interface](#terminal-chat-interface)
* [Session Persistence](#session-persistence)
* [Logging & Analytics](#logging--analytics)
* [Future Enhancements](#future-enhancements)

---

## Installation

```bash
git clone <your-repo-url>
cd kerliixmate_backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch transformers sentencepiece fastapi uvicorn datasets scikit-learn
```

---

## Dataset

Kerliixmate uses a `.jsonl` dataset with seed examples. Each entry has:

```json
{
  "instruction": "How do I reset my password?",
  "response": "You can reset your password at https://kerliix.com/reset-password",
  "category": "account"
}
```

* Dataset file: `datasets/kerliixmate_seed.jsonl`
* Add, update, or delete entries using the **admin tools**.

---

## Training / Fine-Tuning

To train or fine-tune Kerliixmate:

```bash
python kerliixmate_training.py
```

* Uses `google/flan-t5-small` by default.
* Splits dataset into train/validation sets.
* Saves model to `models/kerliixmate_model`.
* Can be retrained anytime with updated datasets.

---

## Running the API

Start the FastAPI backend:

```bash
uvicorn api/kerliixmate_api_model.py --reload --host 0.0.0.0 --port 8000
```

### Endpoints

* `GET /` – Health check and welcome message.
* `POST /chat` – Send user message:

```json
{
  "session_id": "optional-session-id",
  "message": "I forgot my password"
}
```

**Response:**

```json
{
  "session_id": "generated-or-existing-session-id",
  "response": "No worries! You can reset your password at https://kerliix.com/reset-password."
}
```

---

## Admin Tools

* `admin_tools/manage_dataset.py` – Add, delete, or list dataset entries.
* `admin_tools/view_sessions.py` – View active sessions and conversation history.
* `admin_tools/retrain_model.py` – Trigger the fine-tuning pipeline with updated datasets.

---

## Terminal Chat Interface

Test Kerliixmate locally in the terminal:

```bash
python kerliixmate_terminal.py
```

* Supports sessionless chat or persistent sessions with embedding-based or fine-tuned model.
* Useful for testing before API deployment.

---

## Session Persistence

* Supports multi-turn conversation via `session_id`.
* Stores last N exchanges in memory (`MAX_HISTORY`).
* Optional: switch to Redis or database for persistent storage.

---

## Logging & Analytics

* All interactions are logged to `logs/kerliixmate_logs.jsonl` with:

  * `timestamp`
  * `session_id`
  * `user_message`
  * `assistant_response`
* Logs can be analyzed to improve responses, monitor usage, or collect user feedback.

---

## Future Enhancements

* Web-based **admin dashboard** with analytics.
* Multi-user concurrency support and database-backed sessions.
* Enhanced **fine-tuning** and dynamic dataset updates.
* Role-based access control for admin operations.
* Frontend chat integration (React, mobile apps).

---

## License

Kerliixmate is **proprietary to Kerliix Corporation**. All rights reserved.

---

**Kerliixmate** – Your intelligent AI assistant for smooth customer support and engagement.
