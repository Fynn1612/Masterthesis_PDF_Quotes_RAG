import hashlib
import json
import os
# --------------------
# CONFIG
# --------------------
HASH_STORE = r"G:\Meine Ablage\Masterarbeit_RAG_PDFs\hash_store.json"

# --------------------
# UTIL FUNKTIONEN
# --------------------
def compute_md5(file_path):
    """Berechne MD5 Hash einer Datei."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_hashes():
    """Lade gespeicherte Datei-Hashes."""
    if os.path.exists(HASH_STORE):
        with open(HASH_STORE, "r") as f:
            return json.load(f)
    return {}

def save_hashes(hashes):
    """Speichere Datei-Hashes."""
    with open(HASH_STORE, "w") as f:
        json.dump(hashes, f, indent=2)
