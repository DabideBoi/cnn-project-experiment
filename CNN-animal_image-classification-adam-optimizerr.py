from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras.utils as image

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

optim = ['adam', 'nadam', 'adamax', 'adagrad', 'rmsprop']
a_func = ['selu', 'relu', 'sigmoid', 'tanh', 'elu']

for opt in optim:
    for  output in a_func:
        for hidden in a_func:
            classifier = Sequential()
            classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = hidden))
            classifier.add(MaxPooling2D(pool_size = (2, 2)))
            classifier.add(Flatten())
            classifier.add(Dense(units = 128, activation = hidden))
            classifier.add(Dense(units = 1, activation = output))

            classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
            train_datagen = ImageDataGenerator(rescale = 1./255,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True)
            test_datagen = ImageDataGenerator(rescale = 1./255)
            training_set = train_datagen.flow_from_directory('training_set',
                                                            target_size = (64, 64),
                                                            batch_size = 32,
                                                            class_mode = 'binary')
            test_set = test_datagen.flow_from_directory('test_set',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')
            classifier.fit(training_set,
                            steps_per_epoch = (8000/32),
                            epochs = 25 ,
                            validation_data = test_set,
                            validation_steps = (2000/32))   

            scores = classifier.evaluate_generator(test_set, steps=(2000/32))
            accuracy = ("%.2f%%" % (scores[1]*100))
            loss =  ("%.2f" % scores[0])

            with open("results.csv", "a") as myfile:
                myfile.write(str(opt + " " + hidden + " " + output + " " + accuracy + " " + loss + "\n"))




