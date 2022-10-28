import cv2
import face_recognition

img_to_dect = cv2.imread('images/testing/trump-modi.jpg')

while True:

    cv2.imshow('test', img_to_dect)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break


all_face_loc = face_recognition.face_locations(img_to_dect, model= 'hog')

# num_face = len(all_face_loc)

# print(f'There are {num_face} faces in the image')


for index, current_face_location in enumerate(all_face_loc):
    
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print(f"image {index + 1} at ")
    print(f"top : {top_pos}, right : {right_pos}, bottom : {bottom_pos}, left : {left_pos}")
