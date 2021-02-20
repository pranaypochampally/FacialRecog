import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Input, Dropout, Flatten, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.optimizers import  Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from livelossplot.inputs.tf_keras import PlotLossesCallback
print(tensorflow.__version__)

img_size = 48
batch_size = 64
path="/data/"
#creating image data generators for training and validation
datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator = datagen_train.flow_from_directory(path+"train/", target_size=(img_size,img_size), color_mode="grayscale", batch_size=batch_size, class_mode="categorical", shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_train.flow_from_directory(path+"test/", target_size=(img_size,img_size), color_mode="grayscale", batch_size=batch_size, class_mode="categorical", shuffle=False)

model = Sequential()
# convolution Layer-1
model.add(Conv2D(64, (3,3), padding='same', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#convolution Layer-2
model.add(Conv2D(128, (5,5), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#convolution Layer-3
model.add(Conv2D(512, (3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Flatten the model 
model.add(Flatten())

#Dense Layer-1
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

#Dense Layer-2
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

#Dense Layer-3
model.add(Dense(7))
model.add(Activation('softmax'))

#optimizer
opt = Adam(lr=0.0005)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

epochs=10
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, mode='auto')

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', save_weights_only=True, mode='max', verbose=1)

callbacks = [PlotLossesCallback(), checkpoint, reduce_lr]

history = model.fit(
	x = train_generator,
	validation_data=validation_generator,
	steps_per_epoch=steps_per_epoch,
	epochs=epochs,
	validation_steps=validation_steps,
  callbacks=callbacks)