import cv2
import os
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "known_faces")
MODEL_PATH = os.path.join(BASE_DIR, "face_model.yml")
CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("ERROR: Haar cascade file not found!")
    print("Expected location:", CASCADE_PATH)
    exit()

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}

current_label = 0

# Read training dataset
for person_name in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    label_names[current_label] = person_name

    print(f"Processing: {person_name}")

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read: {image_name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]

            faces.append(face)
            labels.append(current_label)

    current_label += 1

# Check whether faces were found
if len(faces) == 0:
    print("ERROR: No faces found in training images.")
    exit()

print(f"Found {len(faces)} face samples.")

# Train model
print("Training face recognition model...")

recognizer.train(
    faces,
    np.array(labels)
)

# Save model
recognizer.write(MODEL_PATH)

# Save labels
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

with open(LABELS_PATH, "w") as file:
    for label, name in label_names.items():
        file.write(f"{label},{name}\n")

print()
print("Training completed successfully!")
print("Model saved as:", MODEL_PATH)
print("Labels saved as:", LABELS_PATH)