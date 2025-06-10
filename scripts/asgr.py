import os
import torch
import json
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

# === Super-Resolution Model ===
from basicsr.archs.swinir_arch import SwinIR
from basicsr.utils.download_util import load_file_from_url

# --- Load SwinIR x4 model (for 4x super-resolution) ---
def load_swinir_model():
    model_path = load_file_from_url(
        url='https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/swinir_sr_classical_patch64_x4.pth',
        model_dir='./models'
    )
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    model.load_state_dict(torch.load(model_path), strict=True)
    return model.eval().to(device)

# === Inference helper ===
def apply_super_resolution(img_pil, model):
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor = model(img_tensor)
    sr_img = sr_tensor.squeeze().cpu().clamp(0, 1)
    sr_img_pil = transforms.ToPILImage()(sr_img)
    return sr_img_pil

# === Set device and models ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = MTCNN(keep_all=False, device=device)
face_encoder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
sr_model = load_swinir_model()

transform = transforms.Compose([
    transforms.ToTensor()
])

# === Embedding extraction with AGSR ===
faces_dir = r"C:\Users\abhi1\Desktop\BIOMETRIC\dataa\faces"
embeddings_dict = {}

if not os.path.exists(faces_dir):
    print("❌ Faces directory not found!")
    exit()

for person_name in os.listdir(faces_dir):
    person_folder = os.path.join(faces_dir, person_name)

    if os.path.isdir(person_folder):
        embeddings = []

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            if img_name.endswith((".jpg", ".png")):
                img = Image.open(img_path).convert("RGB")

                # Step 1: Apply Super-Resolution
                sr_img = apply_super_resolution(img, sr_model)

                # Step 2: Face detection
                face = face_detector(sr_img)
                if face is None:
                    print(f"⚠️ No face detected in {img_name}, skipping...")
                    continue

                face = face.unsqueeze(0).to(device)
                embedding = face_encoder(face).detach().cpu().numpy().flatten()
                embeddings.append(embedding)

        if embeddings:
            embeddings_dict[person_name] = np.mean(embeddings, axis=0).tolist()
        else:
            print(f"⚠️ No valid embeddings for {person_name}, skipping...")

# Save embeddings to JSON
os.makedirs("dataa", exist_ok=True)
with open(r"C:\Users\abhi1\Desktop\BIOMETRIC\scripts\dataa\embeddings.json", "w") as f:
    json.dump(embeddings_dict, f, indent=4)

print("✅ Face embeddings saved with super-resolution!")
