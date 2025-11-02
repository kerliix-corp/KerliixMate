from kerliixmate_api_model import conversations

def show_active_sessions():
    print(f"Active sessions: {len(conversations)}")
    for session_id, history in conversations.items():
        print(f"\nSession ID: {session_id}")
        for turn in history:
            print(f"User: {turn['user']}")
            print(f"Assistant: {turn['assistant']}\n")

if __name__ == "__main__":
    show_active_sessions()
