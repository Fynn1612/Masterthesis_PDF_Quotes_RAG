import os
import json

def get_pdf_metadata(folder):
    """
    Extract metadata from PDF files in the specified folder.
    
    Args:
    folder (str): The path to the folder containing PDF files.

    Return:
    dict: A dictionary containing the PDF metadata.
    """
    metadata = {}
    for f in os.listdir(folder):
        if f.endswith(".pdf"):
            metadata[f] = os.path.getmtime(os.path.join(folder, f))
            
    return metadata

def load_stored_metadata(path):
    """
    Load stored PDF metadata from a JSON file.

    Args:
        path (str): The path to the JSON file containing PDF metadata.

    Returns:
        dict: A dictionary containing the loaded PDF metadata.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_metadata(path, metadata):
    """
    Save PDF metadata to a JSON file.

    Args:
        metadata (dict): The PDF metadata to save.
        path (str): The path to the JSON file where metadata will be saved.
    """
    with open(path, "w") as f:
        json.dump(metadata, f)