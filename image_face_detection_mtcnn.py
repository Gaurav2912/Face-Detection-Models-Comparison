#importing the required libraries
import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
import time

list_dir = os.listdir('col')  

images_list = [cv2.imread(f'col/{item}') for item in list_dir ]

tic = time.time()

#create an instance of MTCNN detector
mtcnn_detector = MTCNN()



for ind, img in enumerate(images_list):

    img_new =  cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #detect all face locations using the mtcnn dectector
    all_face_locations = mtcnn_detector.detect_faces(img_new)

    #print the number of faces detected
    print('There are {} no of faces in this image'.format(len(all_face_locations)))


    #looping through the face locations
    for index,current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get the four position values of current face
        x,y,width,height = current_face_location['box']
        #start co-ordinates
        left_x, left_y = x,y
        #end co-ordinates
        right_x, right_y = x+width, y+height
        #printing the location of current face
        print('Found face {} at left_x:{},left_y:{},right_x:{},right_y:{}'.format(index+1,left_x,left_y,right_x,right_y))

        #draw bounding box around the faces
        cv2.rectangle(img,(left_x,left_y),(right_x,right_y),(0,255,0),2)


        
    #show the image
    cv2.rectangle(img,(10,10), (60, 25) , (255, 0, 0), cv2.FILLED)

    cv2.putText(img, 'mtcnn', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255),1)

    # cv2.imshow("faces in image", img)

    cv2.imwrite(f'mtcnn\{ind+1}.png', img)
    #keep the window waiting until we press a key
    cv2.waitKey(0)
    #close all windows
    cv2.destroyAllWindows()


tac= time.time()

print(tac-tic)