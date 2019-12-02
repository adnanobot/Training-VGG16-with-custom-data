"""
    Reference for source code: https://www.youtube.com/watch?v=INaX55V1zpY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=14

"""
# if importing keras directly creates issues because of the tensorflow 2.0
# so importing the following way is working
# import tensorflow.keras as k

import keras
from keras.layers import Dense
from keras import Sequential
from keras.layers import Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

# loading data
train_path = "./Data/train"
test_path =  "./Data/test"
valid_path =  "./Data/valid"

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (224, 224),
                classes = ["Screwdriver", "Eraser"],  batch_size = 10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (224, 224),
                classes = ["Screwdriver", "Eraser"],  batch_size = 10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size = (224, 224),
                classes = ["Screwdriver", "Eraser"],  batch_size = 10)

"""
vgg = VGG16()
# vgg.summary()

type(vgg) # this inported model is a keras.engine.training.Model type not Sequential type

# so convert Model >> Sequential
classifier = Sequential()
for layer in vgg.layers:
    classifier.add(layer)

# removing the final layer, which was responsible for classification
classifier.layers.pop() 

# going through the vgg model layers and not updating the old weights
for layer in classifier.layers:
    layer.trainable = False

# adding new final layer for classifying only screw-driver and pen
classifier.add(Dense(2, activation = 'softmax')) 

classifier.compile(optimizer = Adam(learning_rate = 0.0001), loss='categorical_crossentropy',
                                metrics=['accuracy'])
classifier.fit_generator(train_batches, epochs = 20, steps_per_epoch = 4,
                        validation_data = valid_batches, validation_steps = 4, verbose = 2)
classifier.summary()
"""