# model.py
import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras import layers
from keras.applications import DenseNet121
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Flatten ,BatchNormalization, Dropout, MaxPooling2D

def miniVGG_model(input_shape, num_classes):
    model = Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def densenet_model(input_shape, num_classes):
    model = Sequential()

    # Add the pre-trained DenseNet121 model (excluding the top layer)
    model.add(DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape))

    # Freeze the weights of the pre-trained model
    model.layers[0].trainable = False

    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))

    # Add Global Average Pooling layer
    model.add(layers.GlobalAveragePooling2D())

    # Add a dense layer
    model.add(layers.Dense(256, activation='relu'))

    # Add the final dense layer for classification with softmax activation
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def resnet_model(input_shape, num_classes):
    ResNet50_Layer = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False)
    ResNet50_Layer.trainable = False

    model = Sequential()
    model.add(ResNet50_Layer)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


# Define your existing models here (if any)

def miniVGG_model_Var1(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model


def densenet_model_var(input_shape, num_classes):
    model = Sequential()

    # Add the pre-trained DenseNet121 model (excluding the top layer)
    model.add(DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape))

    # Freeze the weights of the pre-trained model
    model.layers[0].trainable = False

    # Add Conv2D layers with dropout
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.5))

    # Add Global Average Pooling layer
    model.add(GlobalAveragePooling2D())

    # Add a dense layer with dropout
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # Add the final dense layer for classification with softmax activation
    model.add(Dense(num_classes, activation='softmax'))

    return model


