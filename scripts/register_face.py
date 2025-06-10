import os
import torch
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = MTCNN(keep_all=False, device=device)
face_encoder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings_dict = {}

# Path to faces folder
faces_dir = r"C:\Users\abhi1\Desktop\BIOMETRIC\scripts\dataa"

# Iterate through each person's folder
for person_name in os.listdir(faces_dir):
    person_folder = os.path.join(faces_dir, person_name)
    
    if os.path.isdir(person_folder):  # Ensure it's a folder
        embeddings_dict[person_name] = []  # Store all embeddings for this person
        
        # Process each image in the person's folder
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            if img_name.endswith((".jpg", ".png")):
                img = Image.open(img_path).convert("RGB")
                
                # Detect and encode face
                face = face_detector(img)
                if face is not None:
                    face = face.unsqueeze(0).to(device)
                    embedding = face_encoder(face).detach().cpu().numpy().flatten()
                    embeddings_dict[person_name].append(embedding.tolist())

# Save embeddings to JSON
os.makedirs("dataa", exist_ok=True)  # Ensure directory exists
with open("dataa/embeddings.json", "w") as f:
    json.dump(embeddings_dict, f, indent=4)

print("✅ Face embeddings saved successfully!")
