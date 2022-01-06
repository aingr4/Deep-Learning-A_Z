# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:37:08 2019

@author: aingr
"""
# Building CNN

# Importing modules
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3,3), input_shape = (128,128,3), activation = "relu" ))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Improving test set accuracy
classifier.add(Conv2D(32, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3,3), activation = "relu" ))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 64, activation = "relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = "sigmoid"))

# Compiling the CNN
classifier.compile(optimizer = "adam",  loss = "binary_crossentropy", metrics = ["accuracy"])


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=(8000/32),
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=(2000/32))

# Making single predictions on CNN
import numpy as np
from keras.preprocessing import image

test_image_1 = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (128,128))
test_image_1 = image.img_to_array(test_image_1)
test_image_1 = np.expand_dims(test_image_1, 0)

test_image_2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (128,128))
test_image_2 = image.img_to_array(test_image_2)
test_image_2 = np.expand_dims(test_image_2, 0)

test_image_3 = image.load_img('dataset/single_prediction/nico.jpg', target_size = (128,128))
test_image_3 = image.img_to_array(test_image_3)
test_image_3 = np.expand_dims(test_image_3, 0)

test_image_4 = image.load_img('dataset/single_prediction/parker.jpg', target_size = (128,128))
test_image_4 = image.img_to_array(test_image_4)
test_image_4 = np.expand_dims(test_image_4, 0)


result = classifier.predict(test_image_1)
result = classifier.predict(test_image_2)
result = classifier.predict(test_image_3)
result = classifier.predict(test_image_4)

training_set.class_indices

if(result[0][0] == 1):
    prediction = "dog"
else:
    prediction = "cat"






