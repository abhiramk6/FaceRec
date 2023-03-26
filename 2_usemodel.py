import face_recognition as fr
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import time

import pandas
import csv

known_names = []
known_name_encodings = []

with open('names.txt', 'r') as f:
    my_list = eval(f.read())

ls = list(my_list)

# Check if XML file exists
if os.path.exists('known_faces.xml'):
    # Load known faces from XML file
    tree = ET.parse('known_faces.xml')
    root = tree.getroot()

    for child in root:
        known_names.append(child.attrib['name'])
        encoding = np.array([float(x) for x in child.text.split()])
        known_name_encodings.append(encoding)

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = video_capture.read()

    # Find faces in frame
    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)

    #print(known_names)

    # Recognize faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding, tolerance=0.7)
        name = ""
        distance = None

        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]
            distance = face_distances[best_match_index]
            #print(name)

        # Draw box and label for face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        nm = ''
        if name and distance is not None:
            for i in ls:
                if i[0] == name: nm = i[1]
            confidence = int((1 - distance) * 100)
            cv2.putText(frame, f"{nm} ({confidence}%)", (left + 6, bottom - 6), font, 2.0, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display frame
    cv2.imshow('Face Recognition', frame)

    time.sleep(0.3)

    # Stop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
