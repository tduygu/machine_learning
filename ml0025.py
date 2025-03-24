# # Reading emotions
# # https://www.kaggle.com/code/codingloading/reading-emotions-from-faces-fer2013
#
# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
# import keras
# from keras.preprocessing import image
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import cv2
# from keras import regularizers
# from tensorflow.keras.optimizers import Adam
#
# import warnings
# warnings.filterwarnings('ignore')
# %matplotlib inline
#
# train_dir = "/kaggle/input/fer2013/train"
# test_dir = "/kaggle/input/fer2013/test"
#
# categories = os.listdir(train_dir)
#
# image_counts = {category: len(os.listdir(os.path.join(train_dir, category))) for category in categories}
#
# plt.figure(figsize=(10, 5))
# sns.barplot(x=list(image_counts.keys()), y=list(image_counts.values()), palette="viridis")
# plt.xlabel("Emotion Category")
# plt.ylabel("Number of Images")
# plt.title("Number of Images in Each Emotion Category")
# plt.xticks(rotation=45)
# plt.show()
#
# # Data Augmentation and Preprocessing
# train_datagen = ImageDataGenerator(
#     width_shift_range=0.1,           #  shifts the image horizontally by 10% of the total width
#     height_shift_range=0.1,          # shifts the image vertically by 10% of the total height
#     horizontal_flip=True,            # A left-facing car image might be flipped to a right-facing one
#     rescale=1./255,                  #  improving training stability , Faster Convergence
#     validation_split=0.2
# )
#
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# # Creating Data Generators
# train_generator = train_datagen.flow_from_directory(
#     directory=train_dir,
#     target_size=(48, 48),
#     batch_size=64,
#     color_mode="grayscale",
#     class_mode="categorical",
#     subset="training"
# )
#
# validation_generator = train_datagen.flow_from_directory(
#     directory=train_dir,  # Use train_dir for validation
#     target_size=(48, 48),
#     batch_size=64,
#     color_mode="grayscale",
#     class_mode="categorical",
#     subset="validation"
# )
#
# test_generator = test_datagen.flow_from_directory(
#     directory=test_dir,
#     target_size=(48, 48),
#     batch_size=64,
#     color_mode="grayscale",
#     class_mode="categorical"
# )
#
# # Building the CNN Model
#
# model = tf.keras.Sequential([
#     # input layer
#     tf.keras.layers.Input(shape=(48, 48, 1)),  # Input() instead of input_shape in Conv2D
#     tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Dropout(0.25),
#
#     # 1st hidden dense layer
#     tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Dropout(0.25),
#
#     # 2nd hidden dense layer
#     tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu',
#                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Dropout(0.25),
#
#     # 3rd hidden dense layer
#     tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu',
#                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Dropout(0.25),
#
#     # Flatten layer
#     tf.keras.layers.Flatten(),
#
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.25),
#
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.25),
#
#     # output layer
#     tf.keras.layers.Dense(7, activation='softmax')
# ])
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer,
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
# model.summary()
#
# # Training the Model
# history = model.fit(x = train_generator,epochs = 50 ,validation_data = validation_generator)
#
# # Plotting Training and Validation Accuracy
# plt.plot(history.history["accuracy"], label="Training Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.title("Model Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()
#
# # Evaluating the Model
# test_loss, test_acc = model.evaluate(test_generator)
# print(f"Test Accuracy: {test_acc * 100:.2f}%")
#
