
import torch
import numpy as np
import json
import cv2
import csv
from datetime import datetime
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from face_detection import detect_faces  # Import face detection

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_recognizer = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Load stored embeddings
def load_embeddings():
    try:
        with open(r"C:\Users\abhi1\Desktop\BIOMETRIC\scripts\dataa\embeddings.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("No embeddings found! Run the script to register faces first.")
        return {}

# Extract embedding from face image
def get_face_embedding(face_pil):
    transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
    face_tensor = transform(face_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = face_recognizer(face_tensor)
    
    return embedding.cpu().numpy().flatten()

# Recognize face by comparing with stored embeddings
def recognize_face(face_pil):
    known_faces = load_embeddings()
    unknown_embedding = get_face_embedding(face_pil)

    best_match = None
    best_score = float("inf")

    for name, embedding in known_faces.items():
        dist = np.linalg.norm(np.array(embedding) - unknown_embedding)
        if dist < best_score and dist < 0.8:  # Threshold for face matching
            best_match = name
            best_score = dist

    return best_match if best_match else "Unknown"

# Log attendance
# Keep track of already logged names in this session
logged_names = set()

def log_attendance(name):
    if name in logged_names:
        return  # Don't log again if already logged

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = "dataa/attendance.csv"
    
    try:
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            # Write header only if file is empty
            if file.tell() == 0:
                writer.writerow(["Name", "Timestamp"])
            
            writer.writerow([name, timestamp])
            logged_names.add(name)  # Mark as logged
        
        print(f"✅ Logged: {name} at {timestamp}")
    
    except Exception as e:
        print(f"❌ Error writing to CSV: {e}")


# Start Face Recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera frame not captured. Retrying...")
        continue  # Skip this loop iteration and try again

    faces = detect_faces(frame)  # Detect faces

    if not faces:  
        print("⚠ No faces detected. Waiting for input...")
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue  # Skip the rest of the loop and try again

    for face, (x1, y1, x2, y2) in faces:
        if face is None or face.size == 0:
            print("⚠ Warning: Empty face detected. Skipping...")
            continue  # Skip empty face detections

        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Recognize face
        name = recognize_face(face_pil)

        if name != "Unknown":
            log_attendance(name)  # Log only recognized faces

        # Draw bounding box & name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
