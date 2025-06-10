# import torch
# import numpy as np
# import json
# import cv2
# import csv
# import threading
# from datetime import datetime, timedelta
# from torchvision import transforms
# from PIL import Image
# from facenet_pytorch import InceptionResnetV1
# from face_detection import detect_faces  # Import face detection

# # Initialize model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# face_recognizer = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# # Load stored embeddings
# def load_embeddings():
#     try:
#         with open(r"C:\Users\abhi1\Desktop\BIOMETRIC\scripts\dataa\embeddings.json", "r") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         print("No embeddings found! Run the script to register faces first.")
#         return {}

# # Extract embedding from face image
# def get_face_embedding(face_pil):
#     transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
#     face_tensor = transform(face_pil).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         embedding = face_recognizer(face_tensor)
    
#     return embedding.cpu().numpy().flatten()

# # Recognize face by comparing with stored embeddings
# def recognize_face(face_pil):
#     known_faces = load_embeddings()
#     unknown_embedding = get_face_embedding(face_pil)

#     best_match = None
#     best_score = float("inf")

#     for name, embedding in known_faces.items():
#         dist = np.linalg.norm(np.array(embedding) - unknown_embedding)
#         if dist < best_score and dist < 0.8:  # Threshold for face matching
#             best_match = name
#             best_score = dist

#     return best_match if best_match else "Unknown"

# # Store timestamps for exit handling
# exit_timestamps = {}
# logged_entries = set()

# # Log attendance
# def log_attendance(name, mode):
#     global exit_timestamps, logged_entries
#     timestamp = datetime.now()
#     file_path = f"dataa/{mode.lower()}_attendance.csv"  # entry_attendance.csv or exit_attendance.csv
    
#     if mode == "Entry":
#         if name in logged_entries:
#             return  # Entry is logged only once
#         logged_entries.add(name)
    
#     if mode == "Exit":
#         if name in exit_timestamps:
#             entry_time = exit_timestamps[name]
#             if (timestamp - entry_time).seconds <= 300:
#                 print(f"⏳ {name} returned within 5 minutes. Exit not recorded.")
#                 return  # Ignore short breaks
#         exit_timestamps[name] = timestamp  # Update latest exit time
    
#     try:
#         with open(file_path, mode="a", newline="") as file:
#             writer = csv.writer(file)
            
#             if file.tell() == 0:
#                 writer.writerow(["Name", "Timestamp"])
            
#             writer.writerow([name, timestamp.strftime("%Y-%m-%d %H:%M:%S")])
#         print(f"✅ {mode} Logged: {name} at {timestamp}")
    
#     except Exception as e:
#         print(f"❌ Error writing to CSV: {e}")

# # Process each camera
# def process_camera(camera_source, mode):
#     cap = cv2.VideoCapture(camera_source)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"❌ {mode} Camera not responding. Retrying...")
#             cap.release()
#             cv2.waitKey(2000)
#             cap = cv2.VideoCapture(camera_source)
#             continue
        
#         faces = detect_faces(frame)
#         if not faces:
#             cv2.imshow(f"{mode} Camera", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#             continue
        
#         for face, (x1, y1, x2, y2) in faces:
#             if face is None or face.size == 0:
#                 continue
            
#             face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
#             name = recognize_face(face_pil)
            
#             if name != "Unknown":
#                 log_attendance(name, mode)
            
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
#         cv2.imshow(f"{mode} Camera", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Start Entry and Exit Cameras
# entry_thread = threading.Thread(target=process_camera, args=(0, "Entry"))  # Laptop Webcam
# exit_thread = threading.Thread(target=process_camera, args=("http://192.168.229.24:8080/video", "Exit"))  # Phone Camera

# entry_thread.start()
# exit_thread.start()

# entry_thread.join()
# exit_thread.join()




import torch
import numpy as np
import json
import cv2
import csv
import threading
from datetime import datetime, timedelta
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from face_detection import detect_faces  # Your custom face detection module

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

# Attendance tracking
logged_entries = set()
exit_timestamps = {}
final_exit_logged = set()

# Log attendance (Entry or Exit)
def log_attendance(name, mode):
    global exit_timestamps, logged_entries, final_exit_logged
    timestamp = datetime.now()
    file_path = f"dataa/{mode.lower()}_attendance.csv"

    if mode == "Entry":
        if name in logged_entries:
            return  # Already logged entry
        logged_entries.add(name)
        print(f"✅ Entry Logged: {name} at {timestamp}")
        final_exit_logged.discard(name)  # Reset exit status if re-entering
        try:
            with open(file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(["Name", "Timestamp"])
                writer.writerow([name, timestamp.strftime("%Y-%m-%d %H:%M:%S")])
        except Exception as e:
            print(f"❌ Error writing to CSV: {e}")

    elif mode == "Exit":
        if name in final_exit_logged:
            return  # Exit already logged permanently

        if name in exit_timestamps:
            last_exit_time = exit_timestamps[name]
            if (timestamp - last_exit_time).seconds <= 300:
                print(f"⏳ {name} returned within 5 minutes. Exit not recorded.")
                return  # Short break, ignore
            else:
                final_exit_logged.add(name)  # Mark final exit after 5 mins
        else:
            # First exit detection — record timestamp and wait for 5 min confirmation
            exit_timestamps[name] = timestamp
            print(f"🕒 Exit detected for {name}, waiting 5 min confirmation...")
            return

        # Log exit permanently
        try:
            with open(file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(["Name", "Timestamp"])
                writer.writerow([name, timestamp.strftime("%Y-%m-%d %H:%M:%S")])
            print(f"✅ Exit Logged: {name} at {timestamp}")
        except Exception as e:
            print(f"❌ Error writing to CSV: {e}")

# Process each camera
def process_camera(camera_source, mode):
    cap = cv2.VideoCapture(camera_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ {mode} Camera not responding. Retrying...")
            cap.release()
            cv2.waitKey(2000)
            cap = cv2.VideoCapture(camera_source)
            continue

        faces = detect_faces(frame)
        if not faces:
            cv2.imshow(f"{mode} Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        for face, (x1, y1, x2, y2) in faces:
            if face is None or face.size == 0:
                continue

            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            name = recognize_face(face_pil)

            if name != "Unknown":
                log_attendance(name, mode)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(f"{mode} Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start Entry and Exit Cameras in separate threads
entry_thread = threading.Thread(target=process_camera, args=(0, "Entry"))  # Laptop Webcam
exit_thread = threading.Thread(target=process_camera, args=("http://192.168.143.232:8080/video", "Exit"))  # Phone Camera

entry_thread.start()
exit_thread.start()

entry_thread.join()
exit_thread.join()
