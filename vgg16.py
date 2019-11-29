import keras
from keras.layers import Dense
from keras import Sequential
from keras.layers import Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

# train_path = 
# test_path = 
# valid_path = 

vgg = keras.applications.vgg16.VGG16()
# vgg.summary()

type(vgg) # this inported model is a keras.engine.training.Model type not Sequential type

# so Model >> Sequential
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

classifier.compile(optimizer = optimizers.Adam(learning_rate = 0.0001), loss='categorical_crossentropy',
                                metrics=['accuracy'])
classifier.fit_generator(batch_size = 20, epochs = 5, steps_per_epoch = 4,
                        validatation_data = valid_batches, validation_steps = 4, verbose = 2)
classifier.summary()