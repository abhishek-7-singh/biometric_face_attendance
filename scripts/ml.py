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
# from face_detection import detect_faces  # Custom face detection
# from sklearn.ensemble import RandomForestClassifier
# import os

# # ------------------- ML Model Setup -------------------
# # Pre-trained classifier from earlier training
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(
#     [[55,0],[50,1],[60,0],[30,2],[58,0],[42,1],[62,0],[10,3],[59,1],[49,1],[35,2],[20,3],[57,0],[61,0]],
#     [1,1,1,0,1,0,1,0,1,1,0,0,1,1]
# )

# # ------------------- Face Recognition Setup -------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# face_recognizer = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# def load_embeddings():
#     try:
#         with open("dataa/embeddings.json", "r") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         print("No embeddings found! Register faces first.")
#         return {}

# def get_face_embedding(face_pil):
#     transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
#     face_tensor = transform(face_pil).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = face_recognizer(face_tensor)
#     return embedding.cpu().numpy().flatten()

# def recognize_face(face_pil):
#     known_faces = load_embeddings()
#     unknown_embedding = get_face_embedding(face_pil)
#     best_match, best_score = None, float("inf")
#     for name, embedding in known_faces.items():
#         dist = np.linalg.norm(np.array(embedding) - unknown_embedding)
#         if dist < best_score and dist < 0.8:
#             best_match, best_score = name, dist
#     return best_match if best_match else "Unknown"

# # ------------------- Attendance Tracking -------------------
# entry_logs = {}
# break_logs = {}
# class_duration = timedelta(minutes=3)
# final_marked = set()

# def log_entry(name):
#     if name not in entry_logs:
#         entry_logs[name] = datetime.now()
#         print(f"✅ Entry logged: {name}")
#         append_csv("dataa/entry_attendance.csv", [name, entry_logs[name].strftime("%Y-%m-%d %H:%M:%S")])

# def log_exit(name):
#     now = datetime.now()
#     if name not in entry_logs:
#         print(f"⛔ Exit before entry: {name}")
#         return

#     duration = now - entry_logs[name]

#     # If already marked, ignore
#     if name in final_marked:
#         return

#     # Detect short break
#     if duration < class_duration:
#         break_logs[name] = now
#         print(f"🚻 {name} may be on break. Waiting...")
#         return

#     # Final exit case (for breaks)
#     print(f"🚪 {name} exiting for break.")
#     append_csv("dataa/exit_attendance.csv", [name, now.strftime("%Y-%m-%d %H:%M:%S")])

# def log_return_from_break(name):
#     now = datetime.now()
#     if name in break_logs and (now - break_logs[name]).seconds <= 30:
#         # Person returned within 5 minutes
#         break_logs.pop(name, None)
#         print(f"⏳ {name} returned from break.")
#         log_entry(name)
#     else:
#         # If the break time expired, mark absent
#         print(f"🚪 {name} left for break but didn’t return in time. Marking absent.")
#         mark_final_attendance(name, present=False)

# def mark_final_attendance(name, present):
#     final_marked.add(name)
#     status = "Present" if present else "Absent"
#     print(f"📌 Final Attendance: {name} - {status}")
#     append_csv(r"C:\Users\abhi1\Desktop\BIOMETRIC\scripts\dataa\final_attendance.csv", [name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status])

# def append_csv(path, row):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     write_header = not os.path.exists(path)
#     with open(path, "a", newline="") as f:
#         writer = csv.writer(f)
#         if write_header:
#             if "final" in path:
#                 writer.writerow(["Name", "Timestamp", "Status"])
#             else:
#                 writer.writerow(["Name", "Timestamp"])
#         writer.writerow(row)

# # ------------------- Camera Logic -------------------
# def process_camera(camera_source, mode):
#     cap = cv2.VideoCapture(camera_source)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"❌ {mode} Camera not responding. Reconnecting...")
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
#                 if mode == "Entry":
#                     if name in break_logs:  # Log entry only if the person is returning from break
#                         log_return_from_break(name)
#                     else:
#                         log_entry(name)  # Regular entry logging
#                 elif mode == "Exit":
#                     log_exit(name)  # Log exit when the person is leaving for a break

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         cv2.imshow(f"{mode} Camera", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # ------------------- Start Threads -------------------
# entry_thread = threading.Thread(target=process_camera, args=(0, "Entry"))  # Entry camera
# exit_thread = threading.Thread(target=process_camera, args=("http://192.168.229.24:8080/video", "Exit"))  # Exit camera

# entry_thread.start()
# exit_thread.start()

# entry_thread.join()
# exit_thread.join()

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
# from face_detection import detect_faces  # Custom face detection
# from sklearn.ensemble import RandomForestClassifier
# import os

# # ------------------- ML Model Setup -------------------
# # Pre-trained classifier from earlier training
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(
#     [[55,0],[50,1],[60,0],[30,2],[58,0],[42,1],[62,0],[10,3],[59,1],[49,1],[35,2],[20,3],[57,0],[61,0]],
#     [1,1,1,0,1,0,1,0,1,1,0,0,1,1]
# )

# # ------------------- Face Recognition Setup -------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# face_recognizer = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# def load_embeddings():
#     try:
#         with open("dataa/embeddings.json", "r") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         print("No embeddings found! Register faces first.")
#         return {}

# def get_face_embedding(face_pil):
#     transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
#     face_tensor = transform(face_pil).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = face_recognizer(face_tensor)
#     return embedding.cpu().numpy().flatten()

# def recognize_face(face_pil):
#     known_faces = load_embeddings()
#     unknown_embedding = get_face_embedding(face_pil)
#     best_match, best_score = None, float("inf")
#     for name, embedding in known_faces.items():
#         dist = np.linalg.norm(np.array(embedding) - unknown_embedding)
#         if dist < best_score and dist < 0.8:
#             best_match, best_score = name, dist
#     return best_match if best_match else "Unknown"

# # ------------------- Attendance Tracking -------------------/////////////////////////////////////////////////////////////////////////////////////////////////
# entry_logs = {}
# break_logs = {}
# class_duration = timedelta(minutes=3)
# final_marked = set()

# def log_entry(name):
#     if name not in entry_logs:
#         entry_logs[name] = datetime.now()
#         print(f"✅ Entry logged: {name}")
#         append_csv("dataa/entry_attendance.csv", [name, entry_logs[name].strftime("%Y-%m-%d %H:%M:%S")])

# def log_exit(name):
#     now = datetime.now()
#     if name not in entry_logs:
#         print(f"⛔ Exit before entry: {name}")
#         return

#     duration = now - entry_logs[name]

#     # If already marked, ignore
#     if name in final_marked:
#         return

#     # Detect short break
#     if duration < class_duration:
#         break_logs[name] = now
#         print(f"🚻 {name} may be on break. Waiting...")
#         return

#     # Final exit case (for breaks)
#     print(f"🚪 {name} exiting for break.")
#     append_csv("dataa/exit_attendance.csv", [name, now.strftime("%Y-%m-%d %H:%M:%S")])

# def log_return_from_break(name):
#     now = datetime.now()
#     if name in break_logs and (now - break_logs[name]).seconds <= 300:
#         # Person returned within 5 minutes
#         break_logs.pop(name, None)
#         print(f"⏳ {name} returned from break.")
#         log_entry(name)
#     else:
#         # If the break time expired, mark absent
#         print(f"🚪 {name} left for break but didn’t return in time. Marking absent.")
#         mark_final_attendance(name, present=False)

# def mark_final_attendance(name, present):
#     final_marked.add(name)
#     status = "Present" if present else "Absent"
#     print(f"📌 Final Attendance: {name} - {status}")
#     append_csv("dataa/final_attendance.csv", [name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status])

# def append_csv(path, row):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     write_header = not os.path.exists(path)
#     with open(path, "a", newline="") as f:
#         writer = csv.writer(f)
#         if write_header:
#             if "final" in path:
#                 writer.writerow(["Name", "Timestamp", "Status"])
#             else:
#                 writer.writerow(["Name", "Timestamp"])
#         writer.writerow(row)

# # ------------------- Camera Logic -------------------
# def process_camera(camera_source, mode):
#     cap = cv2.VideoCapture(camera_source)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"❌ {mode} Camera not responding. Reconnecting...")
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
#                 if mode == "Entry":
#                     if name in break_logs:  # Log entry only if the person is returning from break
#                         log_return_from_break(name)
#                     else:
#                         log_entry(name)  # Regular entry logging
#                 elif mode == "Exit":
#                     log_exit(name)  # Log exit when the person is leaving for a break

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         cv2.imshow(f"{mode} Camera", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # ------------------- Start Threads -------------------
# entry_thread = threading.Thread(target=process_camera, args=(0, "Entry"))  # Entry camera
# exit_thread = threading.Thread(target=process_camera, args=("http://192.168.229.24:8080/video", "Exit"))  # Exit camera

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
from face_detection import detect_faces  # Custom face detection
from sklearn.ensemble import RandomForestClassifier
import os

# ------------------- ML Model Setup -------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(
    [[55,0],[50,1],[60,0],[30,2],[58,0],[42,1],[62,0],[10,3],[59,1],[49,1],[35,2],[20,3],[57,0],[61,0]],
    [1,1,1,0,1,0,1,0,1,1,0,0,1,1]
)

# ------------------- Face Recognition Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_recognizer = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def load_embeddings():
    try:
        with open("dataa/embeddings.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("No embeddings found! Register faces first.")
        return {}

def get_face_embedding(face_pil):
    transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
    face_tensor = transform(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = face_recognizer(face_tensor)
    return embedding.cpu().numpy().flatten()

def recognize_face(face_pil):
    known_faces = load_embeddings()
    unknown_embedding = get_face_embedding(face_pil)
    best_match, best_score = None, float("inf")
    for name, embedding in known_faces.items():
        dist = np.linalg.norm(np.array(embedding) - unknown_embedding)
        if dist < best_score and dist < 0.8:
            best_match, best_score = name, dist
    return best_match if best_match else "Unknown"

# ------------------- Attendance Tracking -------------------
entry_logs = {}
break_logs = {}
class_duration = timedelta(minutes=3)
final_marked = set()

def log_entry(name):
    if name not in entry_logs:
        entry_logs[name] = datetime.now()
        print(f"✅ Entry logged: {name}")
        append_csv("dataa/entry_attendance.csv", [name, entry_logs[name].strftime("%Y-%m-%d %H:%M:%S")])

def log_exit(name):
    now = datetime.now()
    if name not in entry_logs:
        print(f"⛔ Exit before entry: {name}")
        return

    duration = now - entry_logs[name]

    if name in final_marked:
        return

    if duration < class_duration:
        break_logs[name] = now
        print(f"🚻 {name} may be on break. Waiting...")
        return

    print(f"🚪 {name} exiting for break.")
    append_csv("dataa/exit_attendance.csv", [name, now.strftime("%Y-%m-%d %H:%M:%S")])

def log_return_from_break(name):
    now = datetime.now()
    if name in break_logs and (now - break_logs[name]).seconds <= 30:  # 5 minutes
        break_logs.pop(name, None)
        print(f"⏳ {name} returned from break.")
        log_entry(name)
    else:
        print(f"🚪 {name} left for break but didn’t return in time. Marking absent.")
        mark_final_attendance(name, present=False)

def mark_final_attendance(name, present):
    final_marked.add(name)
    status = "Present" if present else "Absent"
    print(f"📌 Final Attendance: {name} - {status}")
    append_csv(r"C:\Users\abhi1\Desktop\BIOMETRIC\scripts\dataa\final_attendance.csv", [name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status])

def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            if "final" in path:
                writer.writerow(["Name", "Timestamp", "Status"])
            else:
                writer.writerow(["Name", "Timestamp"])
        writer.writerow(row)

# ------------------- Cooldown Management -------------------
cooldown_period = timedelta(seconds=10)
recently_seen_entry = {}
recently_seen_exit = {}

def is_on_cooldown(name, mode):
    now = datetime.now()
    cooldown_dict = recently_seen_entry if mode == "Entry" else recently_seen_exit
    last_seen = cooldown_dict.get(name)
    if last_seen and (now - last_seen) < cooldown_period:
        return True
    cooldown_dict[name] = now
    return False

# ------------------- Camera Logic -------------------
def process_camera(camera_source, mode):
    cap = cv2.VideoCapture(camera_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ {mode} Camera not responding. Reconnecting...")
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
                if is_on_cooldown(name, mode):
                    continue  # Skip due to cooldown

                if mode == "Entry":
                    if name in break_logs:
                        log_return_from_break(name)
                    else:
                        log_entry(name)
                elif mode == "Exit":
                    log_exit(name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(f"{mode} Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------- Start Threads -------------------
entry_thread = threading.Thread(target=process_camera, args=(0, "Entry"))  # Entry camera
exit_thread = threading.Thread(target=process_camera, args=("http://192.168.229.24:8080/video", "Exit"))  # Exit camera

entry_thread.start()
exit_thread.start()

entry_thread.join()
exit_thread.join()
