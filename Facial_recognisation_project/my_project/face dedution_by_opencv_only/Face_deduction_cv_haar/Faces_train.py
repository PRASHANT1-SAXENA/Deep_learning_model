import os
import numpy as np
import cv2 as cv
import sys

people= []
for i in os.listdir(r'C:\Users\chemi\Desktop\Data_For_face_recognistion\Face_deduction_training'):
    people.append(i)

dir= r"C:\Users\chemi\Desktop\Data_For_face_recognistion\Face_deduction_training"


features=[]
labels=[]
def create_train():
    for i in people:
        path= os.path.join(dir,i)
        label= people.index(i)
        


        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            image_array=cv.imread(img_path)
            scaled_image=cv.cvtColor(image_array,cv.COLOR_BGR2GRAY)
            har_Cascade= cv.CascadeClassifier("face_har.xml")
            face_rect= har_Cascade.detectMultiScale(scaled_image,scaleFactor=1.1,minNeighbors=4)
            print(f'no of face found = {face_rect}')



            for x,y,w,h in face_rect:
                rec_cor = cv.rectangle(scaled_image,(x,y),(x+w,y+h),(0,0,255),4)
                features.append(rec_cor)
                labels.append(label)
                cv.imshow('detected_img', rec_cor)

                
                if cv.waitKey(0) or 0xFF==ord("d"):
                    continue
                    




create_train()
print("Training start -----------")
print(len(features))
print(len(labels))


Face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on hte features list and teh labels list

features = np.array(features,dtype='object')
labels=np.array(labels)


Face_recognizer.train(features,labels)
print('training done--------------')
Face_recognizer.save(r'C:\Users\chemi\Desktop\enviroments\myml\1-Notebooks_python\All Scripts\Deep_learning_model\Facial_recognisation_project\my_project\Face_deduction_cv_haar\recong.yml\face_trained.yml',)
np.save(r'C:\Users\chemi\Desktop\enviroments\myml\1-Notebooks_python\All Scripts\Deep_learning_model\Facial_recognisation_project\my_project\Face_deduction_cv_haar\train\features\features.npy',features)
np.save(r'C:\Users\chemi\Desktop\enviroments\myml\1-Notebooks_python\All Scripts\Deep_learning_model\Facial_recognisation_project\my_project\Face_deduction_cv_haar\train\labels\label.npy',labels) 