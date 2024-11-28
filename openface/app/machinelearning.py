
import numpy as np
import cv2
import sklearn 
import pickle
import os
from sklearn.ensemble import VotingClassifier

from django.conf import settings

STATIC_DIR = settings.STATIC_DIR




#face detection
face_detection_model = cv2.dnn.readNetFromCaffe(os.path.join(STATIC_DIR,'./model/deploy.prototxt.txt'),os.path.join(STATIC_DIR,'./model/res10_300x300_ssd_iter_140000.caffemodel'))

#feature extraction
face_feature_model = cv2.dnn.readNetFromTorch(os.path.join(STATIC_DIR,'./model/openface.nn4.small2.v1.t7'))

# face recognition
face_recognition_model = pickle.load(open(os.path.join(STATIC_DIR,'./model/machinelearning_face_identity.pkl'),'rb'))



# %%
def pipeline_model(path):    
    #pipeline model
    
    img = cv2.imread(path)
    image = img.copy()
    h,w = img.shape[:2]
    #detect face
    Image_blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 117, 123), False, False)

    face_detection_model.setInput(Image_blob)
    detections = face_detection_model.forward()
    
    #machine result
    machinelearning_result = dict(face_detect_score=[],
                                  face_name=[],
                                  face_name_score=[],
                                  emotion_name=[],
                                  emotion_name_score=[],count=[])
    count = 1
    if len(detections) > 0:
        for i , confidence in enumerate(detections[0,0,:,2]):
            if confidence > 0.5:
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startx,starty,endx,endy) = box.astype(int)
                cv2.rectangle(image,(startx,starty),(endx,endy),(0,0,255),1)

                #face feature extraction
                face_roi = image[starty:endy,startx:endx].copy()
                face_blob = cv2.dnn.blobFromImage(face_roi,1/255,(96,96),(0,0,0),True,True)
                face_feature_model.setInput(face_blob)
                vectors = face_feature_model.forward()

                #predict identity
                face_name = face_recognition_model.predict(vectors)
                face_score = face_recognition_model.predict_proba(vectors).max()

                #predict emotion
                # emotion_name = emotion_recognition_model.predict(vectors)
                # emotion_score = emotion_recognition_model.predict_proba(vectors).max()

                text_face = f'{face_name[0]} ({face_score*100:.2f}%)'
                # text_emotion = '{} :{}'.format(emotion_name,emotion_score*10)

                cv2.putText(image,text_face,(startx,starty-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                # cv2.putText(img,text_emotion,(startx,starty-40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
              
              
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/process.jpg'),image)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/roi_{}.jpg'.format(count)),face_roi)
                
                
                machinelearning_result['count'].append(count)
                machinelearning_result['face_detect_score'].append(confidence)  
                machinelearning_result['face_name'].append(face_name)
                machinelearning_result['face_name_score'].append(face_score)
                # machinelearning_result['emotion_name'] = emotion_name
                # machinelearning_result['emotion_name_score'] = emotion_score
                count += 1
           
    return machinelearning_result        
        



