import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

# Load known faces
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"
unknown_faces_dir = "Unknown_faces"

# Create unknown faces directory if it doesn't exist
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)

# Load known faces from both directories
for directory in [known_faces_dir, unknown_faces_dir]:
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = face_recognition.load_image_file(f"{directory}/{filename}")
                encoding = face_recognition.face_encodings(img)
                if encoding:  # Ensure a face was found
                    known_face_encodings.append(encoding[0])
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)

# Counter for unknown faces
stranger_count = 0
last_save_time = {}  # Track when we last saved each face location

# Get existing stranger count
if os.path.exists(unknown_faces_dir):
    existing_strangers = [f for f in os.listdir(unknown_faces_dir) if f.startswith("stranger")]
    if existing_strangers:
        stranger_numbers = []
        for f in existing_strangers:
            try:
                num = int(f.split('stranger')[1].split('.')[0].strip())
                stranger_numbers.append(num)
            except:
                continue
        stranger_count = max(stranger_numbers) if stranger_numbers else 0

def save_unknown_face(frame, face_location):
    global stranger_count
    import time
    
    # Check if we recently saved a face at this location (prevent duplicates)
    current_time = time.time()
    face_key = f"{face_location[0]}-{face_location[1]}-{face_location[2]}-{face_location[3]}"
    
    if face_key in last_save_time:
        if current_time - last_save_time[face_key] < 5:  # 5 second cooldown
            return f"stranger {stranger_count}"  # Return last saved name
    
    stranger_count += 1
    last_save_time[face_key] = current_time
    
    # Extract face region
    top, right, bottom, left = [v * 4 for v in face_location]
    
    # Add some padding around the face
    padding = 20
    face_image = frame[max(0, top-padding):bottom+padding, max(0, left-padding):right+padding]
    
    # Save the face
    filename = f"stranger {stranger_count}.jpg"
    filepath = os.path.join(unknown_faces_dir, filename)
    cv2.imwrite(filepath, face_image)
    
    # Add to known faces for immediate recognition
    rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_face)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(f"stranger {stranger_count}")
    
    return f"stranger {stranger_count}"

# Start webcam
video = cv2.VideoCapture(0)

# Set window properties
cv2.namedWindow("Face Recognition System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition System", 1200, 800)

# Colors for the GUI
PRIMARY_COLOR = (255, 165, 0)  # Orange
SECONDARY_COLOR = (0, 255, 255)  # Cyan
TEXT_COLOR = (255, 255, 255)  # White
BACKGROUND_COLOR = (30, 30, 30)  # Dark gray

# GUI elements
def draw_header(frame, detected_faces):
    height, width = frame.shape[:2]
    # Create header background
    cv2.rectangle(frame, (0, 0), (width, 80), BACKGROUND_COLOR, -1)
    
    # Title
    cv2.putText(frame, "FACE RECOGNITION SYSTEM", (20, 30), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, PRIMARY_COLOR, 2)
    
    # Status info
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {current_time}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
    
    cv2.putText(frame, f"Faces Detected: {detected_faces}", (width - 200, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, SECONDARY_COLOR, 1)

def draw_face_box(frame, face_location, name, confidence):
    top, right, bottom, left = [v * 4 for v in face_location]
    
    # Choose colors based on face type
    if name.startswith("stranger"):
        box_color = (0, 0, 255)  # Red for strangers
        accent_color = (0, 100, 255)  # Light red
    else:
        box_color = PRIMARY_COLOR  # Orange for known faces
        accent_color = SECONDARY_COLOR  # Cyan
    
    # Draw main rectangle with rounded corners effect
    cv2.rectangle(frame, (left-2, top-2), (right+2, bottom+2), box_color, 3)
    cv2.rectangle(frame, (left, top), (right, bottom), accent_color, 2)
    
    # Name background
    text_width = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0][0]
    cv2.rectangle(frame, (left, top-40), (left + text_width + 20, top), BACKGROUND_COLOR, -1)
    cv2.rectangle(frame, (left, top-40), (left + text_width + 20, top), box_color, 2)
    
    # Name text
    cv2.putText(frame, name, (left + 10, top - 15), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_COLOR, 2)
    
    # Confidence bar
    bar_width = int(confidence * 200)
    cv2.rectangle(frame, (left, bottom + 5), (left + 200, bottom + 20), BACKGROUND_COLOR, -1)
    cv2.rectangle(frame, (left, bottom + 5), (left + bar_width, bottom + 20), box_color, -1)
    cv2.putText(frame, f"{confidence:.2f}", (left + 210, bottom + 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

while True:
    ret, frame = video.read()
    if not ret:
        break
        
    # Resize frame for better display
    height, width = frame.shape[:2]
    if width < 800:
        scale_factor = 800 / width
        frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Faster processing
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    detected_faces = len(face_locations)
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        confidence = 0.0
        is_truly_unknown = True

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if face_distances.size > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]  # Convert distance to confidence
                is_truly_unknown = False
        
        # Only save as stranger if it's truly unknown (not recognized at all)
        if is_truly_unknown and name == "Unknown":
            name = save_unknown_face(frame, face_location)
            confidence = 1.0  # New stranger, full confidence

        draw_face_box(frame, face_location, name, confidence)
    
    # Draw header after processing faces
    draw_header(frame, detected_faces)

    cv2.imshow("Face Recognition System", frame)

    if cv2.waitKey(1) == ord("q") :
        break

video.release()
cv2.destroyAllWindows()
