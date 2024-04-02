import cv2
import numpy  as np
import face_recognition as fr
import sys
import os


import pickle

with open('pickle_files/trained_faces.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

known_face_encodings = loaded_data['known_face_encodings']
known_face_names = loaded_data['known_face_names']


video = cv2.VideoCapture(0)
while True:
    isTrue, frame = video.read()
    rgb_frame = frame[:, :, ::-1]
    fc_locations = fr.face_locations(rgb_frame)
    fc_encodings = fr.face_encodings(rgb_frame, fc_locations)

    for (top, right, bottom, left), face_encoding in zip(fc_locations, fc_encodings):
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = 'unknown'
        
        # Check if there are any matches
        if any(matches):
            fr_distances = fr.face_distance(known_face_encodings, face_encoding)
            match_index = np.argmin(fr_distances)
            if matches[match_index]:
                name = known_face_names[match_index]

        cv2.rectangle(frame, (left-15, top-15), (right+15, bottom +  15), (42, 33,60 ), 2)
        cv2.rectangle(frame, (left-15, bottom - 15), (right+ 15, bottom + 15), (38, 25, 65), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left , bottom +8 + 1), font, 1.0, (123, 186, 180), 1)

    cv2.imshow('face_recognition_system', frame)
    k = cv2.waitKey(1)
    if k == ord("d"):
        #cv2.imwrite("mysave_1.jpg", frame)
        sys.exit('you forcefully closed the window')


video.release()
cv2.destroyAllWindows()
