import cv2
import numpy as np
import streamlit as st

# Load Haar Cascade Classifiers
face_path = r"/Users/kailanaresh/Downloads/A VS CODE/8th - intro to cv2-1/open cv -- practicle/Haarcascades/haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(face_path)
eye_path = r"/Users/kailanaresh/Downloads/A VS CODE/8th - intro to cv2-1/open cv -- practicle/Haarcascades/haarcascade_eye.xml"
eye_classifier = cv2.CascadeClassifier(eye_path)
car_path = r"/Users/kailanaresh/Downloads/A VS CODE/8th - intro to cv2-1/open cv -- practicle/Haarcascades/haarcascade_car.xml"
car_classifier = cv2.CascadeClassifier(car_path)

# Functions for different detections
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return image

def detect_eye(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return image

def detect_car(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_classifier.detectMultiScale(gray, 1.4, 5)
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        frames.append(frame)
    cap.release()
    return frames

# Streamlit UI
st.title("OpenCv Project")
st.sidebar.title("Select Detection Mode")

option = st.sidebar.selectbox(
    "Choose detection mode:",
    ("Detect Face with Image", "Detect Eye with Image", "Detect Live Face", "Detect Live Eye", "Detect Cars with Video")
)

if option == "Detect Face with Image":
    st.title("Face Detection with Image")
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if image_file:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        result_img = detect_face(img)
        st.image(result_img, channels="BGR", use_column_width=True)

elif option == "Detect Eye with Image":
    st.title("Eye Detection with Image")
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if image_file:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        result_img = detect_eye(img)
        st.image(result_img, channels="BGR", use_column_width=True)

elif option == "Detect Live Face":
    st.title("Live Face Detection")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.write("Error: Could not access the camera.")
    else:
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_img = detect_face(frame)
            stframe.image(result_img, channels="BGR", use_column_width=True)
            if st.button('Stop Live Detection'):
                break
        cap.release()

elif option == "Detect Live Eye":
    st.title("Live Eye Detection")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.write("Error: Could not access the camera.")
    else:
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_img = detect_eye(frame)
            stframe.image(result_img, channels="BGR", use_column_width=True)
            if st.button('Stop Live Detection'):
                break
        cap.release()

elif option == "Detect Cars with Video":
    st.title("Car Detection in Video")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        video_path = video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        frames = detect_car(video_path)
        for frame in frames:
            st.image(frame, channels="BGR", use_column_width=True)
            
            
            
            
#python -m streamlit run "/Users/kailanaresh/Downloads/A VS CODE/15th - object tracking, fashion mnist, fruit class/facedetctionstreamlit.py"
            #