# import numpy as np

# def detect_faces(image, face_detector):
#     """
#     Detects faces in an image using MTCNN.
    
#     Args:
#         image (numpy array): RGB image.
#         face_detector (MTCNN): Pretrained face detector.
    
#     Returns:
#         faces (list of numpy arrays): Detected face images.
#         boxes (list of tuples): Bounding boxes of faces.
#     """
#     boxes, _ = face_detector.detect(image)

#     faces = []
#     if boxes is not None:
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box)
#             face = image[y1:y2, x1:x2]
#             faces.append(face)

#     return faces, boxes if boxes is not None else []


# ///////////////////////////////////////////////

import cv2
import torch
from facenet_pytorch import MTCNN
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = MTCNN(keep_all=True, device=device)

def detect_faces(frame):
    """Detect faces using MTCNN and return cropped faces + bounding boxes."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = face_detector.detect(rgb_frame)

    faces = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            faces.append((face, (x1, y1, x2, y2)))

    return faces
