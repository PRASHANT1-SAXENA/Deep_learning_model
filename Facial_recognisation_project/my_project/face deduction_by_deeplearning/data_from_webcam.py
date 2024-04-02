import cv2 as cv
import sys

video=cv.VideoCapture(0)
while True:
    count=1
    isTrue, frame= video.read()
    if frame is None:
         sys.exist("Image is not loaded")
    greyscale= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier("face_har.xml")
    #haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_rect =  haar_cascade.detectMultiScale(greyscale, scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in face_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
        cv.imshow("video",frame)
        k= cv.waitKey(1)
        if k==ord("d"):
            cv.imwrite("mysave_1.jpg",frame)
            sys.exit('you forcefully closed the window')
    


#path = input("enter the path of Vedio")
# video=cv.VideoCapture(0)
# name = input("Please your Enter you name:- ")
# while True:
#     count=1
#     isTrue, frame= video.read()
#     if frame is None:
#         sys.exist("Image is not loaded")
#     greyscale= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     haar_cascade = cv.CascadeClassifier("face_har.xml")
#     face_rect =  haar_cascade.detectMultiScale(greyscale, scaleFactor=1.1,minNeighbors=7) 
    
#     for x,y,w,h in face_rect:
#         deduct= cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
#         cv.imshow("video",deduct)
#         # path=r "C:\Users\chemi\Desktop\enviroments\myml\1-Notebooks_python\All Scripts\Deep_learning_model\Facial_recognisation_project\my_project\face deduction_by_deeplearning\webcam_images"
#         # cv.imwrite(f"{name}_{count}", deduct)
#         count=count+1
#         if count>500:
#             sys.exit("500 image is captured")

# video.release()
# cv.destroyAllWindows()
