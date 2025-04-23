
import gradio as gr
import cv2
import numpy as np
import os
import tempfile
import ffmpeg

# Function to compress the video
def compress_video(input_file):
    # Temporary file to save the compressed video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output.close()

    # Compress the video using FFmpeg
    ffmpeg.input(input_file).output(temp_output.name, vcodec='libx264', crf=28).run()

    return temp_output.name

# Function to detect and blur faces
def blur_faces(input_video):
    # Step 1: Compress the video
    compressed_video = compress_video(input_video.name)

    # Step 2: Load the video and face detection model
    video = cv2.VideoCapture(compressed_video)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Output directory for the video frames
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_file.close()
    out = cv2.VideoWriter(output_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))
    
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Blur the faces
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_face

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    video.release()
    out.release()

    return output_file.name

# Gradio interface
def gradio_interface(input_video):
    output_video = blur_faces(input_video)
    return output_video

iface = gr.Interface(fn=gradio_interface, 
                     inputs=gr.inputs.File(label="Upload your video"),
                     outputs=gr.outputs.File(label="Processed Video"))
iface.launch()
