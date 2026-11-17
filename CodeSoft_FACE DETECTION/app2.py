import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Face Detection & Recognition",
    page_icon="👤",
    layout="centered"
)

st.title("👤 Face Detection & Recognition")
st.write("Upload an image to detect and recognize faces.")

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_model.yml"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "labels.txt"
)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    st.error("Haar cascade file not found.")
    st.stop()

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists(MODEL_PATH):
    st.error("Face model not found. Run train_model.py first.")
    st.stop()

recognizer.read(MODEL_PATH)

# Load labels
labels = {}

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as file:
        for line in file:
            label, name = line.strip().split(",", 1)
            labels[int(label)] = name

# Upload image
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_array = np.array(image)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )

    result = image_array.copy()

    if len(detected_faces) == 0:

        st.warning("No face detected.")

    else:

        st.success(f"{len(detected_faces)} face(s) detected.")

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]

            label, confidence = recognizer.predict(face)

            name = labels.get(label, "Unknown")

            # LBPH confidence: lower is better
            if confidence < 80:
                display_name = name
            else:
                display_name = "Unknown"

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                result,
                f"{display_name}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                result,
                f"Confidence: {confidence:.1f}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    st.image(
        result,
        caption="Recognition Result",
        use_container_width=True
    )

st.markdown("---")
