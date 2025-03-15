import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
from PIL import Image
import json

# Storage paths
TRAINING_DIR = "training_images"
ENCODINGS_FILE = "encodings.json"
os.makedirs(TRAINING_DIR, exist_ok=True)

# Load known encodings
def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "r") as f:
            data = json.load(f)
            return [np.array(enc) for enc in data["encodings"]], data["names"]
    return [], []

known_face_encodings, known_face_names = load_encodings()

# Save new encodings
def save_encodings():
    with open(ENCODINGS_FILE, "w") as f:
        json.dump({"encodings": [enc.tolist() for enc in known_face_encodings], "names": known_face_names}, f)

# Streamlit UI
st.title("Face Recognition App")

option = st.sidebar.radio("Choose an option:", ["Train a new face", "Recognize a face"])

if option == "Train a new face":
    name = st.text_input("Enter person's name")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("Train Face"):
        if name and uploaded_file:
            try:
                # Load image
                image = face_recognition.load_image_file(uploaded_file)
                face_locations = face_recognition.face_locations(image)

                if not face_locations:
                    st.error("No face detected. Try another image.")
                elif len(face_locations) > 1:
                    st.error("Multiple faces detected. Please upload a single face image.")
                else:
                    # Encode face and save
                    encoding = face_recognition.face_encodings(image, face_locations)[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                    save_encodings()

                    # Save the image
                    img_path = os.path.join(TRAINING_DIR, f"{name}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.success(f"Successfully trained {name}!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif option == "Recognize a face":
    uploaded_file = st.file_uploader("Upload an image for recognition", type=["jpg", "jpeg", "png"])

    if uploaded_file and st.button("Recognize"):
        try:
            # Load image
            image = face_recognition.load_image_file(uploaded_file)
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                st.error("No face detected.")
            else:
                face_encodings = face_recognition.face_encodings(image, face_locations)
                image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                results = []
                for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_face_encodings, encoding)
                    name = "Unknown"
                    if True in matches:
                        best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, encoding))
                        name = known_face_names[best_match_index]

                    results.append(name)
                    cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(image_cv, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                st.image(image_cv, channels="BGR")
                st.write(f"Detected faces: {', '.join(results)}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
