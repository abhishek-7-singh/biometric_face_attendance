import cv2
import torch
import pandas as pd
import datetime
from PIL import Image
from scripts.face_detection import detect_faces
from scripts.vit_sr import load_vit_sr, enhance_face
from scripts.face_recognition import recognize_face

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_sr = load_vit_sr(device)

# Load Attendance CSV
attendance_file = "dataa/attendance.csv"
try:
    df = pd.read_csv(attendance_file)
except:
    df = pd.DataFrame(columns=["Name", "Timestamp"])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    for face, box in faces:
        x1, y1, x2, y2 = box

        # Convert face to PIL Image
        face_pil = Image.fromarray(face)

        # Enhance face with ViT-SR
        enhanced_face = enhance_face(face_pil, vit_sr, device)

        # Recognize face
        name = recognize_face(enhanced_face)

        # Log attendance
        if name:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df.loc[len(df)] = [name, timestamp]
            df.to_csv(attendance_file, index=False)

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Biometric Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
