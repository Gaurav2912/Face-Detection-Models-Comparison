import numpy as np
import cv2
import face_recognition


webcam_video_stream = cv2.VideoCapture('images/testing/modi.mp4')

while True:
    
    ret, current_frame = webcam_video_stream.read()

    current_frame_small = cv2.resize(current_frame, (0,0), fx= 0.25, fy= 0.25)

    all_face_location = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample= 2, model= 'hog')

    for index, current_face_location in enumerate(all_face_location):
        
        all_pos = current_face_location

        all_pos = np.array(all_pos) * 4
        
        top_pos =  all_pos[0]
        right_pos = all_pos[1]
        bot_pos =  all_pos[2]
        left_pos =  all_pos[3]
         
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bot_pos), (0,0,255), 2)

    cv2.imshow('webcam video', current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break


webcam_video_stream.release()

cv2. destroyAllWindows()