import os
from dotenv import load_dotenv

# Load environment variables from a .env file into os.environ.
# This allows you to keep API keys and secrets out of your codebase.
load_dotenv()

# Retrieve API keys for different LLM providers from environment variables.
# These are used to authenticate requests to the respective APIs.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define a dictionary of available model providers and their respective models.
# This is used to populate model selection options in the app and to manage provider-specific logic.
MODEL_OPTIONS = {
    "Groq": {
        "playground": "https://console.groq.com/",  # Link to Groq's model playground
        "models": [
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
        ],  # List of available Groq models
    },
    "Gemini": {
        "playground": "https://ai.google.dev",  # Link to Google's Gemini playground
        "models": [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
        ],  # List of available Gemini models
    },
}
