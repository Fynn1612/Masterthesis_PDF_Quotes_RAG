import os
from datetime import datetime
import json


def format_chat_history(chat_history):
    """
    Format the chat history as a string for display or for use in prompts.
    This function joins all previous user questions and AI answers into a readable format.

    Args:
        chat_history (list): The chat history to format. Each entry is a dict with "question" and "answer".

    Returns:
        str: The formatted chat history as a string.
    """
    # Each entry is formatted as "User: ...\nAI: ..."
    return "\n".join(
        f"User: {entry['question']}\nAI: {entry['answer']}" for entry in chat_history
    )


def save_chat_history(chat_history, CHAT_HISTORY_DIR):
    """
    Save the current chat history to a JSON file.
    The filename includes a timestamp to ensure uniqueness and traceability.

    Args:
        chat_history (list): The chat history to save.
        CHAT_HISTORY_DIR (str): The directory where chat histories are stored.
    """
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)  # Ensure the directory exists
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )  # Create a timestamp for the filename
    filename = f"chat_{timestamp}.json"
    path = os.path.join(CHAT_HISTORY_DIR, filename)
    # Save the chat history as a pretty-printed JSON file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)


def list_chat_histories(CHAT_HISTORY_DIR):
    """
    List all saved chat history files in the specified directory.
    This is used to display available chat histories in the sidebar.

    Args:
        CHAT_HISTORY_DIR (str): The directory where chat histories are stored.

    Returns:
        list: A sorted list of saved chat history filenames.
    """
    if not os.path.exists(CHAT_HISTORY_DIR):
        return []
    files = sorted(os.listdir(CHAT_HISTORY_DIR))
    return files


def load_chat_history(filename, CHAT_HISTORY_DIR):
    """
    Load a chat history from a JSON file.
    This allows the user to revisit and continue previous conversations.

    Args:
        filename (str): The name of the chat history file to load.
        CHAT_HISTORY_DIR (str): The directory where chat histories are stored.

    Returns:
        list: The loaded chat history as a list of dicts.
    """
    path = os.path.join(CHAT_HISTORY_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
