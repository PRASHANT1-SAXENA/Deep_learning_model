import numpy as np
import cv2 as cv
import os
import sys


people=[] 
for i in os.listdir(r"C:\Users\chemi\Desktop\Data_For_face_recognistion\Face_deduction_training"):
    people.append(i)

haar_cascade= cv.CascadeClassifier('face_har.xml')

# features=np.load(r'C:\Users\chemi\Desktop\enviroments\myml\1-Notebooks_python\All Scripts\Deep_learning_model\Facial_recognisation_project\my_project\Face_deduction_cv_haar\train\features\features.npy',allow_pickle= True)

# labels=np.load(r'C:\Users\chemi\Desktop\enviroments\myml\1-Notebooks_python\All Scripts\Deep_learning_model\Facial_recognisation_project\my_project\Face_deduction_cv_haar\train\labels\label.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"C:\Users\chemi\Desktop\enviroments\myml\1-Notebooks_python\All Scripts\Deep_learning_model\Facial_recognisation_project\my_project\face dedution_by_opencv_only\Face_deduction_cv_haar\recong.yml\face_trained.yml")

tests=[]
for i in os.listdir(r"C:\Users\chemi\Desktop\Data_For_face_recognistion\test"):
    tests.append(i)
for i in tests:
    test=cv.imread(fr"C:\Users\chemi\Desktop\Data_For_face_recognistion\test\{i}")
    test_cvt_scale=cv.cvtColor(test,cv.COLOR_BGR2GRAY)
    #cv.imshow("test_pics", test_cvt_scale)
    
    face_rect=haar_cascade.detectMultiScale(test_cvt_scale,1.1,4)

    for (x,y,w,h) in face_rect:
        faces_roi=test_cvt_scale[y:y+h,x:x+w]
        
        label,confidence=face_recognizer.predict(faces_roi)
        print(f'Label ={people[label]} with a confidence of {confidence}')

        cv.putText(test,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0))
    
        cv.rectangle(test,(x,y),(x+w,y+h),(0,0,255),4)
    cv.imshow("Deducted face",test)
    k = cv.waitKey(0) 
    if k== ord("d"): 
        sys.exit("you finished the testing")

 