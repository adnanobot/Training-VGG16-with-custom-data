"""
    Reference for source code: https://www.youtube.com/watch?v=INaX55V1zpY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=14


# if importing keras directly creates issues because of the tensorflow 2.0
# so importing the following way is working
# import tensorflow.keras as k
"""
# ----- reading >> processing >> encoding - class labels manually using glob class 

# from glob import glob
# class_path = glob(path + "*")
# print("class....", class_labels)


# label = []
# for f in dir:
#   f = str(f)
#   print(f[f.find('\\')+1: ])
#   label.append(f[f.find('\\')+1: ])



import keras
from keras.layers import Dense
from keras import Sequential
from keras.layers import Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

# path
train_path = "./Data_hackathon/train"
valid_path =  "./Data_hackathon/valid"

# Dat augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

train_batches = datagen.flow_from_directory(
    directory = train_path,
    target_size= (224, 224),
    color_mode= 'rgb',
    batch_size=20,
    class_mode='categorical',
    shuffle=True,
    seed=1
)

valid_batches = datagen.flow_from_directory(
    directory = valid_path,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=20,
    class_mode='categorical',
    shuffle=True,
    seed=1
)

print("class indices: ", train_batches.class_indices)


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

# adding new final layer for classifying 6 types of plugs # before it was only screw-driver and pen
classifier.add(Dense(6, activation = 'softmax')) # was classifier.add(Dense(6, activation = 'softmax')) 

classifier.compile(optimizer = Adam(learning_rate = 0.0001), loss='categorical_crossentropy',
                                metrics=['accuracy'])
classifier.fit_generator(train_batches, epochs = 20, steps_per_epoch = 4,
                        validation_data = valid_batches, validation_steps = 4, verbose = 2)
classifier.summary()
