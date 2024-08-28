import cv2
import numpy as np
import dlib
import pygetwindow as gw
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import time
import threading
import speech_recognition as sr

print("Starting Proctoring System...")

# Load YOLO model files
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the facial landmarks predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# Global variables to hold data
switch_data = []
timestamps = []
current_window = None
switch_count = 0
voice_detected = False
head_movement_detected = False
recording = True
cheat_percent = 0.0

# Function to estimate distance from the camera
def estimate_distance(width):
    # Placeholder function: replace with actual distance estimation logic
    return 1.0 / width

# Function to monitor window switches
def monitor_window_switches():
    global current_window, switch_count
    try:
        new_window = gw.getActiveWindow()
        if new_window and new_window != current_window:
            current_window = new_window
            switch_count += 1
            switch_data.append(switch_count)
            timestamps.append(datetime.now().strftime("%H:%M:%S"))
            update_gui()
    except Exception as e:
        print(f"Error in monitor_window_switches: {e}")
    root.after(1000, monitor_window_switches)

# Function to update the GUI
def update_gui():
    try:
        switch_count_label.config(text=f"Switch Count: {switch_count}")
        voice_status_label.config(text=f"Voice Detected: {voice_detected}")
        head_status_label.config(text=f"Head Movement: {head_movement_detected}")
        distance_label.config(text=f"Cheat Percent: {cheat_percent:.2f}")
        ax.clear()
        ax.plot(timestamps, switch_data, marker='o', linestyle='-')
        ax.set_title("Window Switches Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Switch Count")
        canvas.draw()
    except Exception as e:
        print(f"Error in update_gui: {e}")

# Function to detect voice activity
def detect_voice():
    global voice_detected
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)
            try:
                speech = recognizer.recognize_google(audio)
                voice_detected = True
                print(f"[{datetime.now()}] Speech detected: {speech}")
            except sr.UnknownValueError:
                voice_detected = False
            update_gui()
            time.sleep(1)
        except Exception as e:
            print(f"Error in detect_voice: {e}")

# Function to detect head movement
def detect_head_movement():
    global head_movement_detected, cheat_percent
    cap = cv2.VideoCapture(0)
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if faces:
                for face in faces:
                    landmarks = predictor(gray, face)
                    nose_point = (landmarks.part(30).x, landmarks.part(30).y)
                    chin_point = (landmarks.part(8).x, landmarks.part(8).y)
                    head_movement_detected = True
                    cheat_percent = np.linalg.norm(np.array(nose_point) - np.array(chin_point)) / frame.shape[1]
                    if cheat_percent > 0.2:
                        print("Cheating detected")
            else:
                head_movement_detected = False
            update_gui()
            time.sleep(1)
        except Exception as e:
            print(f"Error in detect_head_movement: {e}")

# Function to detect objects and estimate distance
def detect_objects_and_distance(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = str(classes[class_ids[i]])
            distance_from_camera = estimate_distance(w)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {distance_from_camera:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Function to record video
def record_video():
    global recording
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while recording:
        try:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = detect_objects_and_distance(frame)
            out.write(frame)
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                recording = False
        except Exception as e:
            print(f"Error in record_video: {e}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Initialize the Tkinter GUI
print("Initializing Tkinter GUI...")
root = tk.Tk()
root.title("Proctoring System")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

switch_count_label = ttk.Label(frame, text="Switch Count: 0")
switch_count_label.grid(row=0, column=0, pady=10)

voice_status_label = ttk.Label(frame, text="Voice Detected: False")
voice_status_label.grid(row=1, column=0, pady=10)

head_status_label = ttk.Label(frame, text="Head Movement: False")
head_status_label.grid(row=2, column=0, pady=10)

distance_label = ttk.Label(frame, text="Cheat Percent: 0.00")
distance_label.grid(row=3, column=0, pady=10)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(row=4, column=0, pady=10)

# Start monitoring and GUI loop
print("Starting monitoring...")
root.after(1000, monitor_window_switches)

# Start threads for voice detection, head movement detection, and video recording
print("Starting threads...")
threading.Thread(target=detect_voice, daemon=True).start()
threading.Thread(target=detect_head_movement, daemon=True).start()
threading.Thread(target=record_video, daemon=True).start()

print("Entering Tkinter main loop...")
root.mainloop()

