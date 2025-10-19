import cv2
import numpy as np
import os

# === CONFIGURATION ===
video_path = "test.mov"         # Chemin vers ta vidéo
output_folder = "frames_64x64"   # (optionnel) pour visualiser les images
save_npz_path = "dataset_64x64.npz"  # fichier final sauvegardé

# === Création du dossier d'images ===
os.makedirs(output_folder, exist_ok=True)

# === Lecture de la vidéo ===
cap = cv2.VideoCapture(video_path)
frames = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Conversion en noir et blanc
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Redimensionnement à 64x64
    resized = cv2.resize(gray, (64, 64))

    # 3️⃣ Sauvegarde de l'image (optionnel, pour debug)
    filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
    cv2.imwrite(filename, resized)

    # 4️⃣ Stockage normalisé dans un tableau
    frames.append(resized / 255.0)

    frame_count += 1

cap.release()
print(f"✅ {frame_count} images extraites")

# === Conversion en tableau numpy ===
frames = np.array(frames)  # shape = (n_frames, 64, 64)
print("Taille du tableau des frames :", frames.shape)

# === Création du dataset (image t → image t+1) ===
X = frames[:-1]    # toutes sauf la dernière
Y = frames[1:]     # toutes sauf la première

# Aplatissement en vecteurs (si ton réseau utilise des entrées linéaires)
X_flat = X.reshape(X.shape[0], -1).T   # shape = (4096, n-1)
Y_flat = Y.reshape(Y.shape[0], -1).T   # shape = (4096, n-1)

# === Sauvegarde du dataset ===
np.savez(save_npz_path, X=X_flat, Y=Y_flat)

print(f"💾 Dataset sauvegardé dans '{save_npz_path}'")
print(f"X: {X_flat.shape}, Y: {Y_flat.shape}")
