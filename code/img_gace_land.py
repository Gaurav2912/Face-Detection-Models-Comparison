import face_recognition
from PIL import Image, ImageDraw

face_image = face_recognition.load_image_file("images/testing/trump-modi.jpg")

face_landmarks_list = face_recognition.face_landmarks(face_image)

print(face_landmarks_list)
# print(len(face_landmark))

pil_img = Image.fromarray(face_image)

for face_landmarks in face_landmarks_list:
	
	d = ImageDraw.Draw(pil_img)

	for each_landmak in face_landmarks.keys():

		d.line(face_landmarks[each_landmak], fill = (15, 255, 80), width= 1)

pil_img.show()