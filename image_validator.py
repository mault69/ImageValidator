import torch
import numpy as np
import cv2
import os
import dlib
from PIL import Image
from pathlib import Path
from ultralytics import YOLO  # Utilisation de YOLO
import datetime
import torch.serialization

# 🔹 Chemins des modèles
NODE_DIR = Path(__file__).parent  # Dossier du node
LOG_FILE = NODE_DIR / "log.txt"  # Fichier de log
COMFYUI_MODELS_PATH = Path("models/ultralytics/yolo")  # Chemin des modèles YOLO
LANDMARK_PREDICTOR_PATH = Path("models/dlib/shape_predictor_68_face_landmarks.dat")

# 🔹 Fonction pour écrire les logs
def write_log(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_message = f"{timestamp} {message}\n"
    
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(log_message)
    
    print(log_message, end="")  # Afficher aussi dans la console

write_log("🔄 Initialisation du ImageValidator node...")

# 🔹 Vérifier si le dossier des modèles YOLO existe et lister les modèles disponibles
if not COMFYUI_MODELS_PATH.exists():
    write_log(f"⚠️ Le dossier {COMFYUI_MODELS_PATH} n'existe pas.")
    models_available = []
else:
    write_log(f"📂 Contenu du dossier {COMFYUI_MODELS_PATH} :")
    all_files = os.listdir(COMFYUI_MODELS_PATH)

    if not all_files:
        write_log("⚠️ Le dossier est vide !")
    else:
        for f in all_files:
            write_log(f"   - {f}")

    # 🔹 Vérifier si des modèles `.pt` existent (YOLO utilise des .pt)
    models_available = [f for f in all_files if f.endswith(".pt")]

if models_available:
    write_log("✅ Modèles YOLO trouvés :")
    for model in models_available:
        write_log(f"   - {model}")
else:
    write_log("⚠️ Aucun modèle YOLO (.pt) détecté ! Vérifiez que le fichier est bien dans le bon dossier.")

# 🔹 Chargement du détecteur de visages et landmarks si disponible
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = None

if LANDMARK_PREDICTOR_PATH.exists():
    try:
        landmark_predictor = dlib.shape_predictor(str(LANDMARK_PREDICTOR_PATH))
        write_log("✅ Modèle de landmarks faciaux chargé")
    except Exception as e:
        write_log(f"⚠️ Erreur lors du chargement du modèle Dlib : {e}")
else:
    write_log(f"⚠️ Modèle de landmarks faciaux introuvable ({LANDMARK_PREDICTOR_PATH})")


class ImageValidator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.7, "min": 0, "max": 1}),
                "model_name": (models_available,) if models_available else (["Aucun modèle trouvé"],),
            },
        }
    RETURN_TYPES = ("STRING",)
    #RETURN_TYPES = ("STRING", "IMAGE")  # 🆕 Retourne un texte (rapport) + une image annotée
    FUNCTION = "validate_image"
    CATEGORY = "Validation"


    def load_yolo_model(self, model_name, device_choice="auto"):
        """
        Charge un modèle YOLO avec possibilité de choisir entre CPU et CUDA.
        
        :param model_name: Nom du modèle à charger.
        :param device_choice: "cpu" pour forcer l'exécution sur CPU, "cuda" pour utiliser le GPU, "auto" pour détecter automatiquement.
        :return: Modèle YOLO chargé ou message d'erreur.
        """
        model_path = COMFYUI_MODELS_PATH / model_name

        if not model_path.exists():
            write_log(f"⚠️ Modèle sélectionné '{model_name}' introuvable.")
            return None, f"⚠️ Modèle sélectionné '{model_name}' introuvable."

        try:
            # 🔹 Détection automatique du device si "auto"
            if device_choice == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = device_choice.lower()
            
            # 🔹 Vérifier que CUDA est disponible si sélectionné
            if device == "cuda" and not torch.cuda.is_available():
                write_log("⚠️ CUDA non disponible, passage au CPU.")
                device = "cpu"

            write_log(f"🖥️ Modèle YOLO sera chargé sur : {device.upper()}")

            # 🔹 Vérification du type de checkpoint
            checkpoint = torch.load(str(model_path), map_location=device)  
            write_log(f"🧐 Type de checkpoint : {type(checkpoint)}")
            write_log(f"🧐 Contenu du checkpoint : {checkpoint.keys() if isinstance(checkpoint, dict) else 'Non un dict'}")

            # 🔹 Chargement du modèle YOLO
            model = YOLO(str(model_path))
            model.to(device)
            write_log(f"✅ Modèle YOLO chargé sur {device.upper()} : {model_name}")
            return model, None

        except Exception as e:
            write_log(f"⚠️ Erreur lors du chargement de YOLO : {e}")
            return None, f"⚠️ Erreur lors du chargement de YOLO : {e}"


    def validate_image(self, image, threshold, model_name):
        """
        Validate the image for errors, inconsistencies, object coherence, and human anatomy.
        """
        write_log("🔄 Début de la validation de l'image...")
        img = np.array(image)
        report = []  # 📝 Liste des messages du rapport final
        
        # 🔹 Vérification du format de l'image
        if img is None or not isinstance(img, np.ndarray):
            report.append("❌ Erreur : L'image est invalide ou corrompue.")
            write_log(report[-1])
            return ("\n".join(report),)

        write_log(f"ℹ️ Format de l'image : {img.shape}")
        
        # 🔹 Gestion des dimensions (batch, alpha)
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = np.squeeze(img, axis=0)
            write_log(f"🔄 Dimension batch supprimée, nouvelle img.shape = {img.shape}")

        if img.shape[-1] == 4:  
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            write_log("🔄 Image RGBA convertie en RGB")

        # 🔹 Vérification du flou
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 🔹 Vérification et correction du format pour OpenCV
            if gray.dtype != np.uint8:
                write_log("⚠️ Conversion forcée en uint8 pour éviter les erreurs OpenCV.")
                gray = (gray * 255).astype(np.uint8)  # Assurer l'échelle correcte

            # 🔹 Vérification des dimensions
            if len(gray.shape) != 2:  # S'assurer que l'image est bien grayscale
                write_log("⚠️ Problème détecté : L'image ne semble pas être en niveaux de gris.")
                gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

            # 🔹 Calcul du flou après correction
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            write_log(f"✅ Score de flou calculé : {blur_score}")
            
            if blur_score < threshold * 100:
                report.append("⚠️ Image trop floue ou bruitée.")
        except Exception as e:
            report.append(f"❌ Erreur lors de l'analyse du flou : {e}")

        # 🔹 Analyse de colorimétrie
        write_log("🔍 Analyse de colorimétrie en cours...")

        # 🔹 Vérifier que l'image est bien en uint8 (évite les erreurs de colorimétrie)
        if img.dtype != np.uint8:
            write_log("⚠️ Conversion forcée en uint8 pour éviter les erreurs de colorimétrie.")
            img = (img * 255).astype(np.uint8)  # Remettre l'échelle correcte (0-255)

        # 🔹 Vérifier si l'image est en RGB ou BGR avant conversion
        if img.shape[-1] == 3:  # Assurer que l'image est bien en 3 canaux
            write_log("✅ Image confirmée comme RGB/BGR avant conversion HSV.")

        # 🔹 Conversion en HSV pour analyse des couleurs
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # 🔹 Moyenne des canaux HSV
        hue_mean = np.mean(hsv_img[:, :, 0])
        saturation_mean = np.mean(hsv_img[:, :, 1])
        brightness_mean = np.mean(hsv_img[:, :, 2])
        contrast = np.std(img)  # Calcul du contraste
        
        write_log(f"ℹ️ Moyenne de la teinte : {hue_mean}")
        write_log(f"ℹ️ Moyenne de la saturation : {saturation_mean}")
        write_log(f"ℹ️ Moyenne de la luminosité : {brightness_mean}")
        write_log(f"ℹ️ Contraste de l'image : {contrast}")
        
        # 🔹 Moyenne des canaux Rouge, Vert et Bleu
        red_mean = np.mean(img[:, :, 0])  # Rouge
        green_mean = np.mean(img[:, :, 1])  # Vert
        blue_mean = np.mean(img[:, :, 2])  # Bleu

        write_log(f"ℹ️ Moyenne Rouge: {red_mean}, Vert: {green_mean}, Bleu: {blue_mean}")

        if saturation_mean < 30:
            report.append("⚠️ Image avec des couleurs trop ternes.")
        elif saturation_mean > 220:
            report.append("⚠️ Image avec une saturation excessive.")

        if contrast < 20:
            report.append("⚠️ Image avec un contraste trop faible.")

        # 🔹 Détection des objets avec YOLO
        model, error = self.load_yolo_model(model_name)
        if error:
            report.append(error)
            return ("\n".join(report),)

        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
        device = next(model.model.parameters()).device
        img_tensor = img_tensor.to(device)
        results = model(img_tensor)
        
        # 🔹 Vérification que des objets sont bien détectés
        if isinstance(results, list) and len(results) > 0:
            detected_objects = results[0].boxes.data.cpu().numpy()
        else:
            report.append("⚠️ Aucun objet détecté.")
            return ("\n".join(report),)

        write_log(f"ℹ️ {len(detected_objects)} objets détectés")

        class_names = model.names  # Récupérer les classes du modèle

        object_list = []
        alerts = []

        # 🔹 Correction : Extraction correcte des coordonnées des objets
        for obj in detected_objects:
            x_min, y_min, x_max, y_max, score, class_id = obj[:6]
            class_id = int(class_id)
            class_name = class_names.get(class_id, "Inconnu")
            
            obj_width = int(x_max - x_min)
            obj_height = int(y_max - y_min)

            object_info = {
                "classe_nom": class_name,
                "score": float(score),
                "largeur": obj_width,
                "hauteur": obj_height,
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max),
            }
            object_list.append(object_info)
            write_log(f"🧐 Objet détecté : {object_info}")

            # 🔹 Vérification des tailles des objets
            if obj_width < img.shape[1] * 0.05 or obj_height < img.shape[0] * 0.05:
                alerts.append(f"⚠️ Objet trop petit détecté : {class_name}")
            if obj_width > img.shape[1] * 0.9 or obj_height > img.shape[0] * 0.9:
                alerts.append(f"⚠️ Objet trop grand détecté : {class_name}")
        
        # 🔹 Initialisation de la liste des objets détectés pour éviter les erreurs
        detected_objects = []
        
        # 🔹 Vérification et reformattage des objets détectés
        object_list = []
        for obj in detected_objects:
            if isinstance(obj, dict) and all(k in obj for k in ["nom", "score", "dimensions", "x_min", "y_min", "x_max", "y_max"]):
                width, height = map(int, obj["dimensions"].split("x"))
                object_list.append({
                    "classe_nom": obj["nom"],
                    "score": float(obj["score"]),
                    "largeur": width,
                    "hauteur": height,
                    "x_min": int(obj["x_min"]),
                    "y_min": int(obj["y_min"]),
                    "x_max": int(obj["x_max"]),
                    "y_max": int(obj["y_max"]),
                })
            else:
                write_log(f"⚠️ Objet mal formé détecté et ignoré : {obj}")  # Log pour debug

        if isinstance(results, list) and len(results) > 0:
            detected_objects = results[0].boxes.data.cpu().numpy()
        else:
            report.append("⚠️ Aucun objet détecté.")
            return ("\n".join(report),)

        write_log(f"ℹ️ {len(detected_objects)} objets détectés")
        class_names = model.names

        object_list = []
        alerts = []
        for obj in detected_objects:
            x_min, y_min, x_max, y_max, score, class_id = obj[:6]
            class_id = int(class_id)
            class_name = class_names.get(class_id, "Inconnu")
            obj_width = x_max - x_min
            obj_height = y_max - y_min

            object_info = {
                "nom": class_name,
                "score": float(score),
                "dimensions": f"{int(obj_width)}x{int(obj_height)}"
            }
            object_list.append(object_info)
            write_log(f"🧐 Objet détecté : {object_info}")

            if obj_width < img.shape[1] * 0.05 or obj_height < img.shape[0] * 0.05:
                report.append(f"⚠️ Objet trop petit détecté : {class_name}")
            if obj_width > img.shape[1] * 0.9 or obj_height > img.shape[0] * 0.9:
                report.append(f"⚠️ Objet trop grand détecté : {class_name}")

        # 🔹 Vérification des incohérences anatomiques
        write_log("🔍 Vérification des incohérences anatomiques...")
        anatomy_alerts = []

        for obj in object_list:
            class_name = obj["nom"]

            if class_name == "person":
                # 🔹 Vérification et correction du format de l'image pour Dlib
                if img.dtype != np.uint8:
                    write_log("⚠️ Conversion forcée de l'image en uint8 pour éviter l'erreur Dlib.")
                    img = (img * 255).astype(np.uint8)  # Convertir en échelle de 0 à 255

                if len(img.shape) == 3 and img.shape[-1] == 4:  # Si l'image est en RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    write_log("🔄 Conversion de l'image RGBA en RGB.")

                # 🔹 Conversion en grayscale si nécessaire pour Dlib
                if len(img.shape) == 3 and img.shape[-1] == 3:  # Vérifier que c'est bien RGB
                    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    write_log("✅ Image convertie en grayscale pour Dlib.")
                else:
                    img_gray = img  # Si déjà en grayscale, on l'utilise tel quel

                # 🔹 Détection des visages avec Dlib
                face_detected = face_detector(img_gray)

                face_detected = face_detector(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
                if len(face_detected) == 0:
                    anatomy_alerts.append("⚠️ Aucune reconnaissance de visage par Dlib.")
                else:
                    for face in face_detected:
                        landmarks = landmark_predictor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), face)
                        if landmarks is not None:
                            eyes = [36, 39, 42, 45]
                            mouth = [48, 54]
                            nose = [30]
                            missing_parts = [idx for idx in eyes + mouth + nose if not (0 <= landmarks.part(idx).x < img.shape[1] and 0 <= landmarks.part(idx).y < img.shape[0])]

                            if missing_parts:
                                anatomy_alerts.append(f"⚠️ Visage avec parties manquantes (points {missing_parts})")

            if class_name == "hand" or "finger" in class_name.lower():
                obj_width = obj_width
                obj_height = obj_height
                if obj_width / obj_height > 1.5:
                    anatomy_alerts.append(f"⚠️ Proportions anormales pour une main : {obj}")

            if class_name in ["leg", "foot"] and obj_height < img.shape[0] * 0.15:
                anatomy_alerts.append(f"⚠️ Jambe suspecte détectée : trop courte.")

            if class_name in ["arm", "shoulder"] and obj_height < img.shape[0] * 0.10:
                anatomy_alerts.append(f"⚠️ Bras suspect détecté : trop court.")

        # 🔹 Annoter l’image avec les objets détectés
        image_annotated = img.copy()

        for obj in object_list:
            # Vérification avant d'accéder aux clés
            if all(k in obj for k in ["x_min", "y_min", "x_max", "y_max", "classe_nom", "score"]):
                x_min, y_min = int(obj["x_min"]), int(obj["y_min"])
                x_max, y_max = int(obj["x_max"]), int(obj["y_max"])
                label = f"{obj['classe_nom']} {obj['score']:.2f}"

                # Dessiner le rectangle
                cv2.rectangle(image_annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Ajouter le texte
                cv2.putText(image_annotated, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                write_log(f"⚠️ Objet mal formé détecté et ignoré : {obj}")


        # 🔹 Ajouter les infos générales du rapport sur l’image
        cv2.putText(image_annotated, "Validation de l'image", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image_annotated, f"Objets detectes: {len(object_list)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 🔹 Construction du rapport final
        rapport_final = ["📋 Rapport final :"]

        # 🔹 Ajout du score de flou
        rapport_final.append(f"✅ Score de flou : {blur_score:.2f}")

        # 🔹 Ajout de la colorimétrie
        rapport_final.append(f"ℹ️ Moyenne Teinte : {hue_mean:.2f}")
        rapport_final.append(f"ℹ️ Moyenne Saturation : {saturation_mean:.2f}")
        rapport_final.append(f"ℹ️ Moyenne Luminosité : {brightness_mean:.2f}")
        rapport_final.append(f"ℹ️ Contraste : {contrast:.2f}")
        rapport_final.append(f"ℹ️ Moyenne Rouge : {red_mean:.2f}, Vert : {green_mean:.2f}, Bleu : {blue_mean:.2f}")

        # 🔹 Ajout des alertes sur les objets suspects
        if alerts:
            rapport_final.append("⚠️ Anomalies détectées :")
            rapport_final.extend(alerts)

        # 🔹 Ajout des alertes anatomiques
        if anatomy_alerts:
            rapport_final.append("⚠️ Anomalies anatomiques détectées :")
            rapport_final.extend(anatomy_alerts)
        else:
            rapport_final.append("✅ Anatomie correcte.")

        # 🔹 Ajout des objets détectés
        if object_list:
            rapport_final.append(f"ℹ️ {len(object_list)} objets détectés")  # ✅ Garder cette ligne
            for obj in object_list:
                # 🔹 Vérifie que les clés sont bien présentes avant d'ajouter au rapport
                if all(key in obj for key in ["nom", "score", "dimensions"]):
                    rapport_final.append(f"🧐 Objet : {obj['nom']} | Score : {obj['score']:.2f} | Dimensions : {obj['dimensions']}")
                else:
                    write_log(f"⚠️ Objet mal formé détecté : {obj}")  # ✅ Debug log
        else:
            rapport_final.append("⚠️ Aucun objet valide détecté.")  # ✅ Message si aucun objet

        # 🔹 Ajout du rapport aux logs
        for ligne in rapport_final:
            write_log(ligne)
        """
        # Vérifier si l’image a une dimension batch (1, H, W, C) et la supprimer
        if len(image_annotated.shape) == 4 and image_annotated.shape[0] == 1:
            image_annotated = np.squeeze(image_annotated, axis=0)

        # Vérifier si l'image est en niveaux de gris et la convertir en RGB
        if len(image_annotated.shape) == 3 and image_annotated.shape[-1] == 1:
            image_annotated = cv2.cvtColor(image_annotated, cv2.COLOR_GRAY2RGB)
        elif len(image_annotated.shape) == 2:  # Image en noir et blanc
            image_annotated = cv2.cvtColor(image_annotated, cv2.COLOR_GRAY2RGB)

        # Vérifier que l'image est bien entre 0 et 255 et convertir en uint8
        if image_annotated.dtype != np.uint8:
            image_annotated = np.clip(image_annotated, 0, 255).astype(np.uint8)

        # 🔹 Vérification stricte de la forme avant conversion en PIL
        if len(image_annotated.shape) != 3 or image_annotated.shape[-1] != 3:
            raise ValueError(f"Format incorrect après correction : {image_annotated.shape}")

        # 🔹 Convertir en `PIL.Image`
        try:
            image_annotated = Image.fromarray(image_annotated)
        except Exception as e:
            raise ValueError(f"Erreur de conversion en PIL : {e} | Forme de l'image : {image_annotated.shape}")

        # 🔹 Convertir en `torch.Tensor` pour ComfyUI
        image_annotated = torch.tensor(np.array(image_annotated)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 🔹 Retourner le rapport et l’image annotée
        return ("\n".join(rapport_final), image_annotated)
        """
        
        # 🔹 Retourner tout le rapport en une seule string
        return ("\n".join(rapport_final),)
        
NODE_CLASS_MAPPINGS = {
    "ImageValidator": ImageValidator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageValidator": "🔍 Image Validator (YOLO, Visages & Jambes)"
}
