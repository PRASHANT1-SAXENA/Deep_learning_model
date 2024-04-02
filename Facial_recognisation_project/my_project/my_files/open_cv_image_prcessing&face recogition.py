import numpy as np
import cv2  as cv
import sys

######## to read the image file
#  
# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\images.jpeg")
# if img is None:
#     sys.exit("image is not there")
# cv.imshow('image', img)

#-------------------------------------------------#

######## to read the vedio
# video=cv.VideoCapture(r"C:\Users\chemi\Desktop\CV_\videos\video_1.mp4")
# while True:
#     isTrue, frame= video.read() # in True a boolean whether frame was sucessfully read or not and frame is a video read by frame by frame
#     cv.imshow("prashant vedio", frame)
#     if cv.waitKey(20) & 0xFF==ord('q'):
#         cv.imwrite("mysave_1.jpg",frame)
#         break

# video.release()
# cv.destroyAllWindows()

#-------------------------------------------------#

######## To make a blank image with desired color

# blank_image= np.zeros((500,500,3),dtype='uint8')
# blank_image[:]=0,255,0 # assign color
# cv.imshow('blank image',blank_image)
# k=cv.waitKey(0) 
# if k==ord('q'):
#     blank_image[300:400,100:200]=255,0,0
#     cv.imwrite('mysave.jpg',blank_image)

 ######## With range in image to assign color

# blank_image= np.zeros((500,500,3),dtype='uint8')
# blank_image[300:400,100:200]=255,0,0
# cv.imshow('blank image_1',blank_image)
# k=cv.waitKey(0) 
# if k==ord('q'):
#     cv.imshow('mysave.jpg',blank_image)


#-------------------------------------------------#

########## to make shape such rectangle circle etc..such as line or more

# blank_image= np.zeros((500,500,3),dtype='uint8')
# blank_image[:]=0,255,0
# cv.rectangle( blank_image,(0,0), (250,250), (100,80,100 ), thickness=2)
# cv.circle(blank_image,(blank_image.shape[0]//2,blank_image.shape[1]//2),105,(100,100,100),thickness=cv.FILLED) # can use -1 instead of cv.Filled for same result

# cv.imshow('blank image',blank_image)
# cv.waitKey(5000)


#-------------------------------------------------#
######## To put text on the image

# blank_img= np.zeros((500,500,3),dtype='uint8')
# cv.putText(blank_img,"This image is not black",(130,250),cv.FONT_ITALIC,0.5,(0,233,25),2) # 0.5 is a scale here which heigh and wridth of text from where it start here i.e 130,250, and inplace of cv.font_italic we can use any int number for different fonts and last one 2 is a thickness
# cv.imshow("blank img" , blank_img)
# k=cv.waitKey(0)
# if k==ord("d"):
#     sys.exit("you closed the window")   



#-------------------------------------------------#

######## Basic function


#### 1. convert to img in grayscale

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\images.jpeg")
# grey_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("image_1",grey_img)
# if cv.waitKey(0) or 0xFF==ord("q"):
#     sys.exit()


#### 2. Blur the image

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\images.jpeg")
# cv.imshow("image", img)
# blur_img= cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT) # ksize should be in odd as here we take 7,7
# cv.imshow("Blur img", blur_img)
# if cv.waitKey(0) or 0xFF==ord("q"):
#     sys.exit()

#### 3. Edge Cascade

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\images.jpeg")
# canny= cv.Canny(img,50,40)
# cv.imshow("canny image", canny)
# cv.waitKey(0)

#### 4.To resize the image

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\images.jpeg")
# resize_image=cv.resize(img,(1000,500),interpolation=cv.INTER_AREA) # mostly when we shrink the image we use INTER_AREA and when me large the image we often use INTER_LINEAR or INTER_CUBIC(slower but much heigh quality)
# cv.imshow("image",resize_image)
# cv.waitKey(0)

#### or make a function for resize the image

# def resize_image(frame,scale=0.75):
#     width = frame.shape[1]*scale 
#     height = frame.shape[0]* scale
#     dimension = (width,height)
#     return cv.resize(frame,dimension,interpolation=cv.INTER_AREA)


#### 5.To crop the image

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\images.jpeg")
# cropped=img[50:200,200:400]
# cv.imshow('Cropped image', cropped)
# cv.waitKey(0)


#### 6.Rotaion

# def rotate(image,angle,rotPoint=None):
#     height ,width = img.shape[:2]

#     if rotPoint is None:
#         rotPoint= (height//2,width//2) # position of center

#     rotMat= cv.getRotationMatrix2D(rotPoint,angle,1.0) # scale is 1.0
#     dimension =(width,height) 
#     return  cv.warpAffine(img,rotMat,dimension) 

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\images.jpeg")
# rotated = rotate(img,45)
# cv.imshow("rotated image", rotated)
# cv.waitKey(5000)

#### 7.Flip
# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\images.jpeg")
# flip= cv.flip(img,1) # 1 is horizonttaly and -1 is for both horizontally and vertically
# cv.imshow("flip image", flip)
# cv.waitKey(5000)

#-----------------------------------------# 

######### contour Detection

##### first way

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\four_men_img.jpg")
# grey_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# blur_img=cv.GaussianBlur(grey_img,(5,5),cv.BORDER_DEFAULT)
# canny_img= cv.Canny(blur_img,125,175)
# cv.imshow("simple_image",img)
# cv.imshow("gray_image",grey_img)
# cv.imshow("canny_image",canny_img)
# contours , hierarchies = cv.findContours(canny_img,cv.RETR_LIST,cv.CHAIN_APPROX_NONE) # ex- In case fo line chain_approx_none give all the point of line while chain_approx_none give two point of line first and last , retr list give all the contour tree give in hierarchies and external contours  in retrun we got two thing contours and hierarchies where contours is a list of find countours
# print(len(contours))
# cv.waitKey(5000)



#### second way

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\four_men_img.jpg")
# grey_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # instead of blur and canny  we directly the thersold value in below case the gray scale while below become black
# ret , thresh =cv.threshold(grey_img, 125,255 ,cv.THRESH_BINARY) # it returns two value first threshold value is the same which i give i.e 150 and thresh which is binary image or thresholded image # here 150 is the threshold value and and 255 is the value in which i want to change the value which is above then 150
# cv.imshow("thresh_image",thresh)

# contours , hierarchies = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

# print(len(contours))

# k=cv.waitKey(0)
# if k==ord('q'):
#     sys.exit()


#-----------------------------------------------#

######## how to make these contours on blank image

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# cv.imshow("simple_img",img)
# grey_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# rat , thres = cv.threshold(grey_img,125,255,cv.THRESH_BINARY)
# print(rat)
# cv.imshow("thres_img", thres)

# countours, hierarchies = cv.findContours(thres,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

# blank=np.zeros(img.shape,dtype="uint8")
# cv.imshow("blank_img",blank) 

# cv.drawContours(blank,countours,-1,(255,0,0),2) # -1 for all countors
# cv.imshow("countor_img",blank)
# cv.waitKey(0)

# cv.waitKey(0)

# Gennerly it is recommeded that use canny method first to find countours if not good result then can go with  threshold method it is simple but in complex situation it may not work well
 
#---------------------------------------#

######## color spaces 

#### 1. BGR convert to grey scale (previously done)

#### 2. BGR to HSV (high)

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# hsv= cv.cvtColor(img,cv.COLOR_BGR2HSV)
# cv.imshow("HSV Image", hsv)
# cv.waitKey()


#### 3. BGR to LAB some time know as(l*a*b)

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
# cv.imshow("lab img", lab)
# cv.waitKey()

# NOTE- in opencv  read color in BGR format but it is only inside the opencv which is not the current system we use to represent colors outside of open cv is RGB which is inverse 

# example:-
# import matplotlib.pyplot as plt
# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# #cv.imshow("image", img)
# plt.imshow(img)
# plt.show()
 
# That why BGR to RGB scale is important

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")

# rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
# cv.imshow("RGB_Image", rgb)
# cv.waitKey()

# NOTE - we transform to img one into other but we can't convert greyscale directly to hsv so we first convert greyscale to BGR then to hsv


#-------------------------------------------#

######## COLOR CHANNEL

#### how to split and merged the three color channel of image 

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# b,g,r = cv.split(img)

# cv.imshow("blue",b)
# cv.imshow("green",g)
# cv.imshow("red",r)

# print(img.shape)
# print(b.shape) # by default it takes 1 in color channel
# print(g.shape)
# print(r.shape)

# merged = cv.merge([b,g,r])
# cv.imshow("merged", merged)

# blank= np.zeros(img.shape[:2],dtype="uint8")
# blue=cv.merge([b,blank,blank])
# green=cv.merge([blank,g,blank])
# red = cv.merge([blank,blank,r])

# cv.imshow("blank", blank)
# cv.imshow("blue", blue)
# cv.imshow("green",green)
# cv.imshow("red", red)

# cv.waitKey()

#-----------------------------------#

########  how to remove noise or can say smooth out

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")

# average_blur = cv.blur(img,(3,3))  # blur on the basis of making grid of 3*3 taking the average of sourding pixel and assign to every grid
# cv.imshow('average', average_blur)

# gaussian_blur =cv.blur(img, (3,3),0) # In place of blue it gives  weitage to surroding pixels and product of that it looks more natural thant average image where 0 is the standard deviation in x direaction   
# cv.imshow('gaussian blur',gaussian_blur) 

# median_blur =cv.medianBlur(img,3) # In this it assumed an integer value as 3*3 and here it take median in place of average and it the most importantly and widely used in computer vision 

# # NOTE:- mostly median blur is not good at 7 and sometime at 5 but it is very effective in removing noise

# bilateral_blur = cv.bilateralFilter(img,10, 35,25 ) # here 10 is the diametre not window kernal size as all in above ,15 is the sigma color means larger value where there is more color in the neighborhood, this will be consider wher the blur is computed, 15  larger value of this space sigma means the pixel further out from the central pixel will influence the blurring calculation means in this  each pixel effect with the calculation of other pixel
# cv.imshow('bilatera_blur', bilateral_blur)

# # NOTE - IMPORTANT:- # the most effictive and sometimes used in the lot of advanced computer vision projects because of how it blurs other tradianally  blurring method blur the image without looking at whether  you are reducing edges in the image or not biletral  blurring applies blurring but retains the edges in the image as well


# cv.imshow('median blur',median_blur) 

# cv.waitKey()

#---------------------------------------------#

######### Advanced BITWISE OPERATION in OPEN CV (and, or, xor, not)

# blank = np.zeros((400,400),dtype="uint8")

# recatangle =cv.rectangle(blank.copy(), (30,30),(370,370), 255,-1 ) # here as it is not the color image so we take only 255 here as we take 400*400 in 2 dimension and -1 to filled the image
# circle= cv.circle(blank.copy(),(200,200),200,255,-1)

# bitwise_and = cv.bitwise_and(recatangle,circle) # basically the intersection of images pixels
# cv.imshow('bitwise and', bitwise_and)   
 

# bitwise_or = cv.bitwise_or(recatangle,circle) # basically the union of images pixels
# cv.imshow('bitwise or', bitwise_or)  

# bitwise_xor = cv.bitwise_xor(recatangle,circle) # basically the  non- intersection of images pixels
# cv.imshow('bitwise xor', bitwise_xor) 

# bitwise_not = cv.bitwise_not(recatangle) # basically it inverse the binary color that is why it only takes one argument of image
# cv.imshow('bitwise not', bitwise_not) 

# cv.imshow("rectange", recatangle)
# cv.imshow('circle',circle)
# cv.waitKey()


#---------------------------------------#

######## MASKING

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")


# blank= np.zeros(img.shape[:2],dtype="uint8") # for masking it is import to take same dimension as image

# mask = cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)

# masked = cv.bitwise_and(img,img,mask=mask)
# cv.imshow("masked", masked)

# cv.imshow("mask",mask)


# cv.waitKey()

#-----------------------------------------------------#

########  Advanced Computing Histogram

# import matplotlib.pyplot as plt 
# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# grey_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow("image_1",grey_img)

# grey_hist= cv.calcHist([grey_img], [0],None, [256],[0,256]) # [0] is the index of channel in grey scale it is only one index i.e grey,None is for mask as we dont use any mask in this case ,[256] is number of bins , [0,256] is the all possible pixel value

# plt.figure()
# plt.title("Grey scale histogram")
# plt.xlabel("Bins")
# plt.ylabel("No of pixels")
# plt.plot(grey_hist)
# plt.xlim([0,256])
# plt.show()

#### histogram for mask only

# import matplotlib.pyplot as plt 
# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# grey_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# blank = np.zeros((img.shape[:2]),dtype="uint8")
# mask_shape =cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
# masked= cv.bitwise_and(grey_img,grey_img,mask=mask_shape)
# cv.imshow("masked", masked)


# grey_hist= cv.calcHist([grey_img], [0],None, [256],[0,256]) # this time we take mask inplace of None

# plt.figure()
# plt.title("Grey scale histogram")
# plt.xlabel("Bins")
# plt.ylabel("No of pixels")
# plt.plot(grey_hist)
# plt.xlim([0,256])
# plt.show()
# cv.waitKey(0)

####  histogram For color

# import matplotlib.pyplot as plt 
# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")

# blank = np.zeros((img.shape[:2]),dtype="uint8")
# mask_shape =cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
# masked= cv.bitwise_and(img,img,mask=mask_shape)
# cv.imshow("masked", masked)

# color_channel=('b','g','r')

# plt.figure()
# plt.title("BGR Image histogram")
# plt.xlabel("Bins")
# plt.ylabel("No of pixels")

# for i, col in enumerate(color_channel):
#     color_hist = cv.calcHist([img],[i],None,[256],[0,256]) # i is index as in opencv ther bgr format so it take 0 index there blue and so on. 
#     plt.plot(color_hist,color=col)
#     plt.xlim([0,256])
# plt.show()

# cv.waitKey(0)

#------------------------------------------------#

######## Advanced Thresholding

#### simple Thresholding

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Grey", gray)

# threshold, thresh= cv.threshold(gray,150,255,cv.THRESH_BINARY )  # it returns two value first threshold value is the same which i give i.e 150 and thresh which is binary image or thresholded image # here 150 is the threshold value and and 255 is the value in which i want to change the value which is above then 150

# cv.imshow("simple threshold image", thresh)
# cv.waitKey(0)

#### thershold inverse

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Grey", gray)

# threshold, thresh_inv= cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)  # it returns two value first threshold value is the same which i give i.e 150 and thresh which is binary image or thresholded image # here 150 is the threshold value and and 255 is the value in which i want to change the value which is above then 150. 

# cv.imshow("simple threshold_inv  image", thresh_inv)
# cv.waitKey(0) 

# Note- in the downside of above threshold wau we have to give theshold value manually which will not work in advance cases so we can do let the computer find the optimal treshold value by itself. this way is called the adaptive threshold.


#### ADAPTIVE THRESHOLD

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Grey", gray) 
# adaptive_thres = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,3  ) # 11 blocksize of the kernel size at which it compute the mean and c = 3 which is the  subtracted from the mean allowing us to fine tune(not that much important) our threshold and can set this to 0.
# cv.imshow("Adaptive thres img", adaptive_thres)
# cv.waitKey()


#### ADAPTIVE THRESHOLD INVERSE

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Grey", gray) 
# adaptive_thres_inv = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,11,3  ) # 11 blocksize of the kernel size at which it compute the mean and c = 3 which is the  subtracted from the mean allowing us to fine tune(not that much important) our threshold and can set this to 0.
# cv.imshow("Adaptive thres img", adaptive_thres_inv)
# cv.waitKey()



#### with gaussian
# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Grey", gray) 
# adaptive_thres = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,3 )
# cv.imshow("Adaptive thres img", adaptive_thres)
# cv.waitKey()


#-------------------------------------------------#

######### ADvanced Edge Dection (Other than canny method)

#### Laplaction

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Grey", gray) 

# lap = cv.Laplacian(gray,cv.CV_64F) # cv.CV_64F is data depth it is gredient base(df/dx) it involve trasition white to blank and vise-versa that consider a positive and negative value and as image does not consider negative number so take a absoute value.

# lap=np.uint8(np.absolute(lap))
# cv.imshow('Laplacian',lap)
# cv.waitKey()

#### sobel 

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\fox_img.jpg")
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Grey", gray) 
# sobelx= cv.Sobel(gray, cv.CV_64F,1,0)
# sobely= cv.Sobel(gray, cv.CV_64F,0,1)

# cv.imshow('soble X', sobelx)
# cv.imshow('soble Y', sobely)
# combine_sobel =cv.bitwise_or(sobelx,sobely)
# cv.imshow('combine sobel',combine_sobel)

# canny = cv.Canny(gray,150,175) # canny is multiprocess somewhere it in the banckend it uses sobel
# cv.imshow('canny',canny)
# cv.waitKey(0)


#----------------------------------------------------#

##### Face deduction in img

# img= cv.imread(r"C:\Users\chemi\Desktop\CV_\Screenshot 2023-08-01 165638.jpg")
# if img is None:
#     sys.exist("Image is not loaded")
# # face deduction is not involved color tone skin or tone so we take grey sclae 
# greyscale= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# #cv.imshow('greyscale',greyscale)  
# haar_cascade = cv.CascadeClassifier("face_har.xml")

# face_rect =  haar_cascade.detectMultiScale(greyscale, scaleFactor=1.1,minNeighbors=7) # the min number of neigbours rectangle should have to be called a face it return the coordinates of the rectangle for face if it is deducting more face then it really has as it is very sesitive so increase the value of minNeighbour increase from 3 to let say 7.

# print(f'the face found = {len(face_rect)}')
# print(face_rect.shape)
# for x,y,w,h in face_rect:
#     deduct= cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
#     cv.imshow('detection', deduct)
    

 
# if cv.waitKey(0) & 0xFF==ord('d'):
#     sys.exit()


###### Face dedcution in vedio

# path = input("enter the path of Vedio")
# video=cv.VideoCapture(r"C:\Users\chemi\Desktop\CV_\videos\video_1.mp4")
# while True:
#     isTrue, frame= video.read() # in True a boolean whether frame was sucessfully read or not and frame is a video read by frame by frame
#     if frame is None:
#         sys.exist("Image is not loaded")
#     # face deduction is not involved color tone skin or tone so we take grey sclae 
#     greyscale= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     #cv.imshow('greyscale',greyscale)  
#     haar_cascade = cv.CascadeClassifier("face_har.xml")

#     face_rect =  haar_cascade.detectMultiScale(greyscale, scaleFactor=1.1,minNeighbors=7) # the min number of neigbours rectangle should have to be called a face it return the coordinates of the rectangle for face if it is deducting more face then it really has as it is very sesitive so increase the value of minNeighbour increase from 3 to let say 7.
#     print(face_rect)
    
#     for x,y,w,h in face_rect:
#         deduct= cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
#         cv.imshow('detection', deduct)

#         if cv.waitKey(20) & 0xFF==ord('q'):
#             cv.imshow("mysave_1.jpg",frame)
#             break

# video.release()
# cv.destroyAllWindows()










 



















 














    


