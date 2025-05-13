# Computer Vision Mini Projects

This repository contains two Python-based computer vision projects: **Air Canvas** and **Sleep Detection**. Both projects use OpenCV and related libraries to demonstrate real-time video processing and interactive applications.

---

## 1. Air Canvas

**Air Canvas** lets you draw in the air using a colored object or hand gestures, tracked via your webcam.

- **Features:**
  - Draw on a digital canvas by moving a colored object in front of your webcam.
  - Real-time video processing with OpenCV.
  - Save your artwork as an image.

- **How to Run:**
  1. Install dependencies:
      ```sh
      pip install opencv-python numpy
      ```
  2. Run the script:
      ```sh
      python AIR CANVAS PROJECT/aircanvas.py
      ```

---

## 2. Sleep Detection

**Sleep Detection** monitors a user's eyes via webcam to detect drowsiness and trigger an alarm if the eyes remain closed for too long.

- **Features:**
  - Detects facial landmarks and monitors eye closure.
  - Plays an alarm sound if sleep/drowsiness is detected.
  - Uses pre-trained dlib model for face and eye detection.

- **How to Run:**
  1. Install dependencies:
      ```sh
      pip install opencv-python dlib imutils scipy
      ```
  2. Ensure `shape_predictor_68_face_landmarks.dat` is present in the `SLEEP DETECTION` folder.
  3. Run the script:
      ```sh
      python SLEEP DETECTION/SleepDetection.py
      ```

---

## Folders

- `AIR CANVAS PROJECT/` — Contains `aircanvas.py` for air drawing.
- `SLEEP DETECTION/` — Contains `SleepDetection.py`, alarm sounds, model file, and output folder.

