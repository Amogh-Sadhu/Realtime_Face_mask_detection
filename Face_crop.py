import os 
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img_path = list(paths.list_images("C:\\Users\\amogh\\anaconda3\\envs\\mini_project\\Dataset\\images\\NoMask"))
j=0
for i in img_path :
    image = cv2.imread(i)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_detect.detectMultiScale(gray, 1.05, 6)
    j = j+1
    for x, y, w, h in face:
        face = image[y:y+h,x:x+w]
        cv2.imwrite("cropped{}.jpg".format(j),face)
    