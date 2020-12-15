import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
import  tensorflow_hub as hub
#Parameters
image_path = "./images/pic1.jpeg"
face_detector_path = "./face_detector"
model_path = "./mask_detector.model"
conf_threshold = 0.5

#Load face detector model
print("Loading face detector model.")
face_modelPath = os.path.sep.join([face_detector_path,"deploy.prototxt"])
face_modelWeigths = os.path.sep.join([face_detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])
face_net = cv2.dnn.readNet(face_modelPath,face_modelWeigths) #Read model with opencv

#Load MobileNet model
print("Face mask detector loading.")
model = tf.keras.models.load_model(model_path)

#Blob from the image
#Applies preprocessing on image before send it to CNN.
#Mean subtraction is used to help combat illumination changes in the input images in our dataset.
image = cv2.imread(image_path)
original_img = image.copy()
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))

print("Detecs faces")
face_net.setInput(blob)
detections = face_net.forward()

#loop over the detections
print(detections.shape) #(1,1,200,7)
print(detections[0][0][0])
for i in range(0,detections.shape[2]):
    #extract probability of detection
    confidence = detections[0,0,i,2]

    #filter detections
    if confidence>conf_threshold:
        #compute x-y coordinates of bounding box from the face detector.
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype("int")

        #ensure bounding boxes fall within the dimensions of frame
        (startX,startY) = (max(0,startX),max(0,startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        #extract the face ROI
        #convert BGR to RGB
        #resize 224x224 -- CONV net input shape

        face = image[startY:endY,startX:endX]
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face = cv2.resize(face,(224,224))
        face = tf.keras.preprocessing.image.img_to_array(face)
        face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
        face = np.expand_dims(face,axis=0)

        #feed the mobile net with the detected face
        (mask,without_mask) = model.predict(face)[0]

        if mask>without_mask: label = "Mask"
        else: label =="No Mask"

        color = (0,255,0) if label == "Mask" else (0,0,255)

        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)


cv2.imshow("Output",image)
cv2.waitKey(0)






