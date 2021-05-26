import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2 
import numpy as np
from PIL import Image

imgpath = input("Enter path of the image file : ")
img = cv2.imread(imgpath)
cv2.imshow('Image',img)

model = load_model("mask_detector.model")

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
resized_img = cv2.resize(img, (224,224))
resized_img = img_to_array(resized_img)
resized_img = np.reshape(resized_img,(-1,224,224,3))
resized_img = preprocess_input(resized_img)
result=model.predict(resized_img)
label=np.argmax(result,axis=1)[0]

if label==1:
    cv2.putText(img,'No mask',(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(img,'Accuracy :{:.1f}%'.format(np.amax(result,axis=1)[0]*100),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
    
elif label==0:
    cv2.putText(img,'Mask',(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(img,'Accuracy :{:.1f}%'.format(np.amax(result,axis=1)[0]*100),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
cv2.imshow('Image',img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


        