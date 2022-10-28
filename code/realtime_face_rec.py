import numpy as np
import cv2
import face_recognition

webcam_video_stream = cv2.VideoCapture(0)

modi_img = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_img)[0]

trump_img = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_img)[0]

# gaurav_img = face_recognition.load_image_file('download.jpg')
# gaurav_face_encodings = face_recognition.face_encodings(gaurav_img)[0]

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = [modi_face_encodings, trump_face_encodings]
known_face_names = ["Narendra Modi", "Donald Trump"]


while True:
    #get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()

    current_frame_small = cv2.resize(current_frame, (0,0), fx= 0.25, fy= 0.25)
    
    all_face_location = face_recognition.face_locations(current_frame_small,
    model= 'hog')

    all_face_encoding = face_recognition.face_encodings(current_frame_small, 
    all_face_location)

    for current_face_location, current_face_encoding in zip(all_face_location, all_face_encoding):
        all_pos = current_face_location

        all_pos = np.array(all_pos) * 4
        
        top_pos =  all_pos[0]
        right_pos = all_pos[1]
        bottom_pos =  all_pos[2]
        left_pos =  all_pos[3]
         
        all_matches = face_recognition.compare_faces(known_face_encodings,
        current_face_encoding)

        name_of_person = 'unknown'

        if True in all_matches:
            index_true = all_matches.index(True)
            name_of_person = known_face_names[index_true]


        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)


    cv2.imshow('webcam',current_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam_video_stream.release()
cv2.destroyAllWindows()  