import csv
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
tf.test.is_gpu_available()
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from sklearn.utils import shuffle

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader( csvfile )
    for line in reader:
        lines.append(line)

folder =  './data/IMG/'
correction = 0.3

def generator(lines, batch_size=32):
    #make data generator
    num_samples = len(lines)

    while 1: # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            images = []
            measurements = []

            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                measurement  = float( batch_sample[3] )
                flipped_measurement = - measurement

                if correction == 0:
                    pic_num = 1
                else:
                    pic_num = 3

                for i in range(pic_num):
                    filename = source_path.split('\\')[-1]
                    current_path = folder + filename
                    image = cv2.imread(current_path)
                    images.append( image )
                    image_flipped = cv2.flip( image,1 )
                    images.append(image_flipped)

                    if i==0:
                        measurements.append(measurement)
                        measurements.append(flipped_measurement)
                    elif i==1:
                        measurements.append(measurement+correction)
                        measurements.append(flipped_measurement+correction)
                    elif i==2 :
                        measurements.append(measurement-correction)
                        measurements.append(flipped_measurement-correction)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import LeakyReLU
train_samples, validation_samples = train_test_split(lines, test_size=0.3)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

input_shape=(160,320,3)

#creat net
model = Sequential()
#normalize input
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
#remove top and bottom part of the image as they are not relevant for the training

model.add(Cropping2D(cropping=((70,20),(0,0))))
model.add( Convolution2D(24,5,5, subsample=(2,2), activation="relu") )
model.add(LeakyReLU( alpha=0.3 ))
model.add( Dropout(0.5) )
model.add( Convolution2D(36,5,5, subsample=(2,2), activation="relu") )
model.add(LeakyReLU( alpha=0.3 ))
model.add( Convolution2D(48,5,5, subsample=(2,2), activation="relu") )
model.add(LeakyReLU( alpha=0.3 ))
model.add( Convolution2D(64,3,3,activation="relu") )
model.add(LeakyReLU( alpha=0.3 ))
model.add( Dropout(0.5) )
model.add( Convolution2D(64,3,3,activation="relu") )
model.add(LeakyReLU( alpha=0.3 ))
model.add( Flatten() )
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=  len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples),
                    nb_epoch=5 )

model.save('model.h5')
print('finished!')
