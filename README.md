# 🧠 Simple Face Recognition with OpenCV

This is a beginner-friendly Python project that uses **OpenCV** and **face_recognition** to detect and recognize faces in real time using a webcam.

## 📸 Features
- Detects faces from webcam feed.
- Recognizes known faces from a folder of images.
- Labels recognized faces with names.
- Automatically saves and recognizes unknown faces as "stranger X".
- Live statistics showing current time and number of detected faces.
- Press `q` to exit.

## 🧰 Requirements
- Python 3.7+
- OpenCV
- face_recognition
- numpy
- dlib (automatically installed with `face_recognition`)

Install dependencies using:

```bash
pip install opencv-python face_recognition numpy
```
## 📁 File Structure

```
face_recognizer/
├── face_recognition_simple.pY    # Main program
├── known_faces/                  # Put known people's photos here
│   
├── Unknown_faces/                # Auto-generated stranger photos
│   
└── README.md                     # This file
```


