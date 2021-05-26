import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2 
import numpy as np

model = load_model("Mask Detection_Amogh")

video_stream = cv2.VideoCapture(0)

while(True):
    ret, img = video_stream.read()

    resized_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(resized_img, (224,224))
    resized_img = img_to_array(resized_img)
    resized_img = np.reshape(resized_img,(-1,224,224,3))
    resized_img = preprocess_input(resized_img)
    result=model.predict(resized_img,)
    label=np.argmax(result,axis=1)[0]
   
    if label==1:
        cv2.rectangle(img,(0,0),(210,70), (186,186,186), -1)
        cv2.putText(img,'No mask',(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.putText(img,'Accuracy :{}%'.format(np.amax(result,axis=1)[0]*100),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        
        
    elif label==0:
        cv2.rectangle(img,(0,0),(210,70),(186,186,186),-1)
        cv2.putText(img,'Mask',(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(img,'Accuracy :{}%'.format(np.amax(result,axis=1)[0]*100),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        
    
    cv2.imshow('Video feed',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


        