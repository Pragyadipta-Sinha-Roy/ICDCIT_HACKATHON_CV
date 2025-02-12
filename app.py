import streamlit as st
import cv2
import numpy as np
from PIL import Image

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

def save_image(image, filename):
    cv2.imwrite(filename, image)

def load_known_faces():
    known_faces = []
    labels = []
    return known_faces, labels

def train_recognizer(known_faces, labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(known_faces, np.array(labels))
    return recognizer

st.title("Face Recognition and Identification")

if st.button("Capture Image"):
    image = capture_image()
    st.image(image, channels="BGR")
    save_image(image, "captured_image.jpg")
    st.success("Image captured and saved!")

if st.button("Identify Face"):
    known_faces, labels = load_known_faces()
    recognizer = train_recognizer(known_faces, labels)
    
    unknown_image = cv2.imread("captured_image.jpg", 0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(unknown_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = unknown_image[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)
        st.write(f"Identified: {label} with confidence {confidence}")
