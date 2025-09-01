import os
from datetime import datetime
import json

def format_chat_history(chat_history):
    """Format the chat history for display.

    Args:
        chat_history (list): The chat history to format.

    Returns:
        str: The formatted chat history.
    """
    # chat_history ist eine Liste von Dicts mit "question" und "answer"
    return "\n".join(
        f"User: {entry['question']}\nAI: {entry['answer']}" for entry in chat_history
    )

def save_chat_history(chat_history, CHAT_HISTORY_DIR):
    """Save the chat history to a JSON file.

    Args:
        chat_history (list): The chat history to save. 
    """
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_{timestamp}.json"
    path = os.path.join(CHAT_HISTORY_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

def list_chat_histories(CHAT_HISTORY_DIR):
    """List all saved chat history files.
    
    Returns:
        list: A list of saved chat history filenames.
    """
    
    if not os.path.exists(CHAT_HISTORY_DIR):
        return []
    files = sorted(os.listdir(CHAT_HISTORY_DIR))
    return files

def load_chat_history(filename, CHAT_HISTORY_DIR):
    """Load chat history from a JSON file.

    Args:
        filename (str): The name of the chat history file to load.

    Returns:
        list: The loaded chat history.
    """
    path = os.path.join(CHAT_HISTORY_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
