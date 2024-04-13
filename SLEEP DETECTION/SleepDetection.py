import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
import time

# Initializing pygame mixer
pygame.mixer.init()

# Loading beep sound files
beep_sound = pygame.mixer.Sound("D:/Project/SLEEP DETECTION/beep.wav")
drowsy_sound = pygame.mixer.Sound("D:/Project/SLEEP DETECTION/alarm.wav")

# Status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Initializing face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("d:/Project/SLEEP DETECTION/shape_predictor_68_face_landmarks.dat")


# Initialize face frame outside the loop
face_frame = None

# Initialize time variables
start_time = time.time()
last_awake_time = time.time()
last_slept_time = time.time()
last_face_detected_time = time.time()  # Initialize last face detected time
last_face_not_detected_time = time.time()  # Initialize last face not detected time

# Initialize total time counters
total_awake_time = 0
total_drowsy_time = 0
total_sleep_time = 0

# Function to compute distance between two points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Function to check if the eyes are closed or not
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0

# Ask user if they want to continue with the default camera index
while True:
    use_default = input("Do you want to continue with the default camera - WEBCAM (index = 0)? (y/n): ").lower()
    if use_default == 'y':
        cap = cv2.VideoCapture(0)  # Use default camera index
        break
    elif use_default == 'n':
        while True:
            try:
                camera_index = int(input("Enter the index number of the camera device: "))
                cap = cv2.VideoCapture(camera_index)  # Use the specified camera index
                break
            except ValueError:
                print("Please enter a valid integer for the camera index.")
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")

if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    faces = detector(gray)

    # Update face frame only when face detected
    if len(faces) > 0:
        last_face_detected_time = time.time()  # Update last face detected time when face detected
        last_face_not_detected_time = time.time()  # Reset last face not detected time
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            face_frame = frame.copy()  # Update face_frame when face detected
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "SLEEPING !!!"
                    color = (0, 0, 255)
                    beep_sound.play()  # Play beep sound when sleeping
                    cv2.circle(face_frame, (landmarks[36][0], landmarks[36][1]), 10, (0, 0, 255), 2)
                    cv2.circle(face_frame, (landmarks[42][0], landmarks[42][1]), 10, (0, 0, 255), 2)

                    total_sleep_time += elapsed_time
                    last_slept_time = time.time()

            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy !"
                    color = (0, 255, 255)
                    drowsy_sound.set_volume(0.5)  # Set volume to 50%
                    drowsy_sound.play()  # Play drowsy sound when drowsy
                    cv2.circle(face_frame, (landmarks[36][0], landmarks[36][1]), 10, (0, 255, 255), 2)
                    cv2.circle(face_frame, (landmarks[42][0], landmarks[42][1]), 10, (0, 0, 255), 2)

                    total_drowsy_time += elapsed_time

            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status = "Active :)"
                    color = (0, 255, 0)

                    total_awake_time += elapsed_time
                    last_awake_time = time.time()

            # Display time awake and drowsy
            cv2.putText(frame, f"Total Awake: {total_awake_time:.2f} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Total Drowsy: {total_drowsy_time:.2f} sec", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Total Sleep: {total_sleep_time:.2f} sec", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.putText(frame, f"Last Slept: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_slept_time))}", (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Last Awake: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_awake_time))}", (frame.shape[1] - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Display status text on the frame
            cv2.putText(face_frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    else:
        # Face not detected, update last face not detected time
        last_face_not_detected_time = time.time()

    # Display last face detected time in red at bottom middle
    cv2.putText(frame, f"Last Face Detected: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_face_detected_time))}", (int(frame.shape[1]/2) - 150, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if face_frame is not None:  # Display face frame only if defined
        cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
