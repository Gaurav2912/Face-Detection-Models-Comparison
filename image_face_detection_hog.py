#importing the required libraries
import cv2
import os
import dlib
import time


list_dir = os.listdir('col')  

#loading the image to detect
images_list = [cv2.imread(f'col/{item}') for item in list_dir]

tic = time.time()

#load the pretrained HOG SVN model
face_detection_classifier = dlib.get_frontal_face_detector()



for ind, img in enumerate(images_list):

    # img_gry =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detect all face locations using the HOG SVN classifier
    all_face_locations = face_detection_classifier(img, 2)

    #print the number of faces detected
    print('There are {} no of faces in this image'.format(len(all_face_locations)))

    #looping through the face locations
    for index,current_face_location in enumerate(all_face_locations):
        #start and end co-ordinates
        left_x, left_y, right_x, right_y = current_face_location.left(),current_face_location.top(),current_face_location.right(),current_face_location.bottom()
        #printing the location of current face
        print('Found face {} at left_x:{},left_y:{},right_x:{},right_y:{}'.format(index+1,left_x,left_y,right_x,right_y))
        #
        cv2.rectangle(img,(left_x,left_y),(right_x,right_y),(0,255,0),2)


    #show the image
    cv2.rectangle(img,(10,10), (50, 25) , (255, 0, 0), cv2.FILLED)

    cv2.putText(img, 'hog', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255),1)
    cv2.imshow(f"faces in {ind}",img)

    # cv2.imwrite(f'hog\{ind+1}.png', img)
    #keep the window waiting until we press a key
    cv2.waitKey(0)
    #close all windows
    cv2.destroyAllWindows()

tac = time.time()


print(tac-tic)