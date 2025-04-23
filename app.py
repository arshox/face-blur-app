import gradio as gr
import cv2
import tempfile
from moviepy.editor import VideoFileClip
import os

def blur_faces_in_video(video_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Temp paths
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    
    with open(temp_input, "wb") as f:
        f.write(video_path.read())

    cap = cv2.VideoCapture(temp_input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_face
        out.write(frame)

    cap.release()
    out.release()

    return temp_output

iface = gr.Interface(
    fn=blur_faces_in_video,
    inputs=gr.Video(label="Upload your video"),
    outputs=gr.Video(label="Blurred video"),
    title="Face Blur Tool",
    description="Upload a video and this tool will blur all detected faces."
)

iface.launch()