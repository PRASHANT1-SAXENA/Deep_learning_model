import os
import caer
import canaro
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import gc
import tensorflow as tf
from tensorflow.keras import layers, models
import sys

image_size= (80,80)
channel =1
char_path=r"C:\Users\chemi\Desktop\Data_For_face_recognistion\Face_deduction_training"

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char]=len(os.listdir(os.path.join(char_path,char)))
#print(char_dict)
char_dict= caer.sort_dict(char_dict,descending=True)
#print(char_dict)

characters = []
for i in char_dict:
    characters.append(i[0])

train = caer.preprocess_from_dir(char_path, characters,channels=channel,IMG_SIZE=image_size,isShuffle=True)
# can you open cv as well 
# plt.figure(figsize=[30,30])
# plt.imshow(train[0][0],cmap="gray")
# plt.show()

feature_set , label = caer.sep_train(train, IMG_SIZE=image_size)
# print(feature_set)
# print(label)

# Normilize data into zero to one

from tensorflow.keras.utils import to_categorical
feature_set= caer.normalize(feature_set)
label= to_categorical(label,len(characters))
# print(feature_set)
# print(label)

x_train,x_val,y_train, y_val= caer.train_val_split(feature_set,label,val_ratio=.2)

del train
del feature_set
del label
gc.collect()

batch_size=32

#optimizer = tf.keras.optimizers.legacy.SGD
datagen= canaro.generators.imageDataGenerator()
train_gen= datagen.flow(x_train,y_train,batch_size= batch_size)

# model = canaro.models.createSimpsonsModel(IMG_SIZE=image_size, channels=channel, output_dim=len(characters), loss='binary_crossentropy', decay=1e-7, learning_rate=0.001, momentum=0.9,nesterov=True) # which is basically length of list of characters here we take only three character

# building model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output to feed into dense layers
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='sigmoid'))  # Using sigmoid for binary classification

learning_rate = 0.001
momentum = 0.9
nesterov = True
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#print(model.summary())

from tensorflow.keras.callbacks import LearningRateScheduler
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)] 

epochs=50
training = model.fit(train_gen,steps_per_epoch=len(x_train)//batch_size, epochs=epochs,validation_data=(x_val,y_val),validation_steps=len(y_val)//batch_size,callbacks=callbacks_list)

def prepare_for_test(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img=cv.resize(img,image_size)
    img=caer.reshape(img,image_size,1)
    return img

img=r"C:\Users\chemi\Desktop\Data_For_face_recognistion\test\yogi ji_14.jpg"
img_array=cv.imread(img)
prediction=model.predict(prepare_for_test(img_array))
print (prediction)







# for char in os.listdir(char_path):
#     char_images_path=os.path.join(char_path,char)
#     for i in os.listdir(char_images_path):
#         test_img_path=os.path.join(char_images_path,i)
#         print(test_img_path)
#         img=cv.imread(test_img_path)
#         cv.imshow("image", img)
#         prediction=model.predict(prepare_for_test(img))
#         print(prediction)
#         print(characters[np.argmax(prediction[0])])
#         k= cv.waitKey(0)
#         if k==ord("d"):
#             sys.exit("you stopped forcefully")













