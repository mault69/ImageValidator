ImageValidator
📌 Présentation
Le ImageValidator est un nœud ComfyUI permettant d'analyser des images en utilisant YOLO (détection d'objets) et Dlib (détection faciale). Il identifie et évalue les objets, analyse les caractéristiques colorimétriques et détecte les anomalies anatomiques dans une image.

Ce node est conçu pour :

Valider la qualité d’une image (flou, contraste, saturation, etc.).
Détecter des objets et analyser leur taille et cohérence (via YOLO).
Analyser les visages et vérifier la présence de caractéristiques faciales cohérentes (via Dlib).
Générer un rapport détaillé de validation sous forme de texte.
📥 Entrées du Node
Le node accepte les paramètres suivants :

Paramètre	Type	Description
image	IMAGE	Image d’entrée à analyser.
threshold	FLOAT	Seuil de tolérance (ex: pour le flou). Valeur entre 0 et 1.
model_name	STRING	Nom du modèle YOLO à utiliser. Liste des modèles disponibles dans models/ultralytics/yolo/.
📤 Sorties du Node
Sortie	Type	Description
rapport	STRING	Rapport détaillé de l’analyse de l’image.
🔹 Note : Initialement, le node devait aussi retourner une image annotée (IMAGE), mais cette option a été temporairement désactivée.

🛠 Fonctionnalités du Node
1️⃣ Chargement des modèles
Le script vérifie la présence de modèles YOLO et Dlib sur le disque.
Il charge automatiquement un modèle YOLO s’il est disponible.
Le détecteur de visages Dlib est chargé si son fichier shape_predictor_68_face_landmarks.dat est trouvé.
2️⃣ Prétraitement de l’image
Vérifie que l’image est un numpy array valide.
Supprime les dimensions inutiles (batch dimension, alpha channel).
Convertit en RGB si nécessaire (si en grayscale ou RGBA).
Vérifie que l’image est bien en uint8 (0-255).
3️⃣ Analyse de la qualité d’image
Flou : Calcule un score de flou via Laplacian variance.
Contraste : Vérifie si l’image a un contraste suffisant.
Saturation : Détecte si l’image est trop terne ou sursaturée.
4️⃣ Détection des objets (YOLO)
Charge le modèle YOLO et l’applique à l’image.
Récupère les objets détectés (nom, confiance, coordonnées, dimensions).
Vérifie si les objets sont trop petits ou trop grands par rapport à l’image.
5️⃣ Analyse des incohérences anatomiques (Dlib)
Si un humain est détecté (class_name == "person") :

Vérifie que le visage contient bien les yeux, le nez et la bouche.
Détecte les mains, bras, jambes et analyse leur proportionnalité.
6️⃣ Génération du rapport final
Le rapport contient :

✅ Score de flou, contraste, saturation
📌 Liste des objets détectés
⚠️ Alertes sur la taille des objets
🔍 Anomalies anatomiques détectées
📌 Exemple d’Exécution
🖼 Entrée :
Image contenant une personne et un objet flou.

📋 Rapport de sortie :
plaintext
Copier
Modifier
📋 Rapport final :
✅ Score de flou : 32.56
ℹ️ Moyenne Teinte : 120.34
ℹ️ Moyenne Saturation : 45.78
ℹ️ Moyenne Luminosité : 180.23
ℹ️ Contraste : 12.67
ℹ️ Moyenne Rouge : 120.12, Vert : 115.34, Bleu : 98.89
⚠️ Image trop floue ou bruitée.
⚠️ Objet trop petit détecté : cup
🧐 Objet : Person | Score : 0.95 | Dimensions : 230x450
🧐 Objet : Cup | Score : 0.80 | Dimensions : 40x50
⚠️ Aucune reconnaissance de visage par Dlib.
