import sys
import os

# Ajouter le dossier actuel aux chemins d'importation
sys.path.append(os.path.dirname(__file__))

# Importer le custom node ImageValidator
from .image_validator import ImageValidator  # Assurez-vous que le fichier s'appelle bien image_validator.py

# Enregistrer le node dans ComfyUI
NODE_CLASS_MAPPINGS = {
    "ImageValidator": ImageValidator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageValidator": "🔍 Image Validator (Visages, Mains & Jambes)"
}
