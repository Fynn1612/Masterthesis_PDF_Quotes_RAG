import os
import json


def get_pdf_metadata(folder):
    """
    Extract metadata from PDF files in the specified folder.
    This function scans the folder for PDF files and records their last modified timestamps.
    This metadata is used to detect new or changed PDFs for (re-)processing.

    Args:
        folder (str): The path to the folder containing PDF files.

    Returns:
        dict: A dictionary mapping each PDF filename to its last modified timestamp.
    """
    metadata = {}
    for f in os.listdir(folder):
        if f.endswith(".pdf"):
            metadata[f] = os.path.getmtime(os.path.join(folder, f))
    return metadata


def load_stored_metadata(path):
    """
    Load stored PDF metadata from a JSON file.
    This is used to remember which PDFs have already been processed and when.

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
    This allows the app to persist information about which PDFs have been processed and their modification times.

    Args:
        path (str): The path to the JSON file where metadata will be saved.
        metadata (dict): The PDF metadata to save.
    """
    with open(path, "w") as f:
        json.dump(metadata, f)
