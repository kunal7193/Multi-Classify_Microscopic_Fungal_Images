#train_evaluate.py

import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np

class EarlyStop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') is not None and logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.95:
            print("\nReached wanted accuracy so cancelling training!")
            self.model.stop_training = True

early_stop_cb = EarlyStop()

class ModelTrainer:
    def __init__(self, model, preprocessed_data, labels):
        self.model = model
        self.preprocessed_data = preprocessed_data
        self.labels = labels

def train_and_evaluate_minivgg(model, train_generator, valid_generator, test_generator, epochs=30): #change the epochs to 30 after testing
    initial_learning_rate = 0.00005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=1000, decay_rate=0.9)
    
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    start_time = time.time()
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // valid_generator.batch_size, callbacks=[early_stopping])
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    # Evaluate model on test set
    test_loss, test_accuracy = model.evaluate(
        test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f"Test Accuracy: {test_accuracy}")

    # Generate predictions for evaluation metrics
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # ROC Curve and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = test_generator.num_classes
    y_test = tf.keras.utils.to_categorical(true_classes, num_classes=n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    return history, predicted_classes, class_labels, fpr, tpr, roc_auc

def train_and_evaluation_densenet(model, train_generator, valid_generator, test_generator, epochs=100): # change epochs to 100 after testing 
    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    start_time = time.time()
    history_densenet = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // valid_generator.batch_size,
        callbacks=[early_stopping])
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    # Evaluate model on test set
    test_loss, test_accuracy = model.evaluate(
        test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f"Test Accuracy: {test_accuracy}")

    # Generate predictions for evaluation metrics
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # ROC Curve and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = test_generator.num_classes
    y_test = tf.keras.utils.to_categorical(true_classes, num_classes=n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    return history_densenet, predictions, true_classes, class_labels, fpr, tpr, roc_auc, cm

def train_and_evaluation_resnet(model, train_generator, valid_generator, test_generator, epochs=40): #change back to 40 after testing
    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // valid_generator.batch_size,
        callbacks=[early_stop_cb])
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    # Evaluate model on test set
    test_loss, test_accuracy = model.evaluate(
        test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f"Test Accuracy: {test_accuracy}")

    # Generate predictions for evaluation metrics
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # ROC Curve and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = test_generator.num_classes
    y_test = tf.keras.utils.to_categorical(true_classes, num_classes=n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    return history, predictions, true_classes, class_labels, fpr, tpr, roc_auc, cm


def train_and_evaluate_miniVGG_Var1(model, train_generator, valid_generator, test_generator, epochs=100):  # change epochs to 100 after testing
    # Define the initial learning rate
    initial_learning_rate = 0.00005

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=1000, decay_rate=0.9)

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    start_time = time.time()
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // valid_generator.batch_size,
        callbacks=[early_stopping])
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    # Evaluate model on test set
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f"Test Accuracy: {test_accuracy}")

    # Generate predictions for evaluation metrics
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # ROC Curve and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = train_generator.num_classes
    y_test = tf.keras.utils.to_categorical(true_classes, num_classes=n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    cm = confusion_matrix(true_classes, predicted_classes)

    return history, predictions, true_classes, class_labels, fpr, tpr, roc_auc, cm


def train_and_evaluate_densenet_var(model, train_generator, valid_generator, test_generator, epochs=100): # change epochs to 100 after testing
    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    start_time = time.time()
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // valid_generator.batch_size,
        callbacks=[early_stopping])
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    # Evaluate model on test set
    test_loss, test_accuracy = model.evaluate(
        test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f"Test Accuracy: {test_accuracy}")

    # Generate predictions for evaluation metrics
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # ROC Curve and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = test_generator.num_classes
    y_test = tf.keras.utils.to_categorical(true_classes, num_classes=n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    return history, predictions, true_classes, class_labels, fpr, tpr, roc_auc, cm



