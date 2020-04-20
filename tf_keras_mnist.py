import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers
import numpy as np

mnits = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnits.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# 堆叠模型

model = tf.keras.Sequential()

model.add(layers.Flatten(input_shape=(28, 28)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32)

model.evaluate(x_test, y_test, verbose=2)