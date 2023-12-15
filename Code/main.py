# main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to suppress all messages
from data_handler import DataHandler
import time
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from model import miniVGG_model , densenet_model , resnet_model , miniVGG_model_Var1, densenet_model_var
from train_evaluate import train_and_evaluate_minivgg , train_and_evaluation_densenet, train_and_evaluation_resnet, train_and_evaluate_miniVGG_Var1, train_and_evaluate_densenet_var
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get the path to the current script
script_path = os.path.abspath(__file__)

# Construct the main folder path relative to the script
main_folder = os.path.join(os.path.dirname(script_path), 'defungi')

subfolders = ['H1', 'H2', 'H3', 'H5', 'H6']

# Data Handling and Splitting
data_handler = DataHandler(main_folder, subfolders)
data_handler.load_and_split_data()

# Print the number of images in each directory
data_handler.print_image_counts()

# Print the number of images in each class for each dataset
data_handler.print_image_counts_by_class()

data_handler = DataHandler(main_folder, subfolders)

# Display one image from each class in the training set
print("Displaying one image from each class in the training set:")
data_handler.display_images(os.path.join(main_folder, 'train'))

train_generator, valid_generator, test_generator = data_handler.create_datagens()

# Choose an image from the test dataset using the DataHandler method
class_name = 'H5'
img_path = data_handler.get_train_image_path(class_name)
if img_path is not None:
    print("Train Image Path:", img_path)

    # Load the original image without rescaling for display
    original_img = load_img(img_path, target_size=(data_handler.img_width, data_handler.img_height))

    # Convert the image to a numpy array
    original_array = img_to_array(original_img)

    # Reshape the array to (1, img_width, img_height, 3) as the generator expects a batch of images
    original_array = original_array.reshape((1,) + original_array.shape)

    # Data normalization and augmentation with multiple transformations
    augmentation_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,  
        fill_mode='nearest'
    )

    # Generate augmented images (applying multiple transformations)
    augmented_images = [augmentation_datagen.random_transform(original_array[0]) for i in range(5)]

    # Plot the original and augmented images
    plt.figure(figsize=(10, 5))

    # Plot original image without rescaling
    plt.subplot(2, 3, 1)
    plt.imshow(original_array[0].astype('uint8'))  # Convert back to uint8 for display
    plt.title('Original Image')
    plt.axis('off')

    # Plot augmented images
    for i in range(5):
        plt.subplot(2, 3, i + 2)
        plt.imshow(augmented_images[i].astype('uint8'))  # Convert back to uint8 for display
        plt.title(f'Augmentation {i+1}')
        plt.axis('off')

    plt.show()
else:
    print(f"No valid image files found for class {class_name} in the test set.")

input("Press Enter to continue...")

# Set input shape and number of classes
input_shape = (128, 128, 3)  # Change the dimensions as per your images
num_classes = len(train_generator.class_indices)


def create_results_folder(model_name):
    script_path = os.path.abspath(__file__)
    results_folder = 'results'
    model_folder = os.path.join(os.path.dirname(script_path), results_folder, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    return model_folder

# Create the "results" folder for MiniVGG model
minivgg_folder = create_results_folder('minivgg_model')
densenet_folder = create_results_folder('densenet_model')
resnet_folder = create_results_folder('resnet_model')
minivgg_var1_folder = create_results_folder('minivgg_var1_model')
densenet_var_folder = create_results_folder('densenet_var_model')

# Train and evaluate MiniVGG model
minivgg_model = miniVGG_model(input_shape, num_classes)
minivgg_model.summary()

# Save the model architecture plot in minivgg_folder
tf.keras.utils.plot_model(minivgg_model, to_file=os.path.join(minivgg_folder, 'minivgg_model_architecture.png'), show_shapes=True)

# Train and evaluate MiniVGG model using the new function
history_minivgg_model, predicted_classes_minivgg_model, class_labels, fpr_minivgg, tpr_minivgg, roc_auc_minivgg = train_and_evaluate_minivgg(minivgg_model, train_generator, valid_generator, test_generator, epochs=30) # Adjust epoch is needed
# Plot training vs validation loss
plt.figure(figsize=(8, 5))
plt.plot(history_minivgg_model.history['loss'], label='Training Loss')
plt.plot(history_minivgg_model.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss for MiniVGG Model')
plt.grid()
plt.savefig(os.path.join(minivgg_folder, 'training_vs_validation_loss.png'))  # Save the plot
plt.show()

# Plot training vs validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history_minivgg_model.history['accuracy'], label='Training Accuracy')
plt.plot(history_minivgg_model.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy for MiniVGG Model')
plt.grid()
plt.savefig(os.path.join(minivgg_folder, 'training_vs_validation_accuracy.png'))  # Save the plot
plt.show()

# Plot confusion matrix
true_classes_minivgg_model = test_generator.classes
cm_minivgg_model = confusion_matrix(true_classes_minivgg_model, predicted_classes_minivgg_model)
plt.figure(figsize=(5, 5))
sns.heatmap(cm_minivgg_model, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for MiniVGG Model')
plt.savefig(os.path.join(minivgg_folder, 'confusion_matrix.png'))  # Save the plot
plt.show()

input("Press Enter to continue...")

num_classes_minivgg_model = len(class_labels)

# Plot ROC curve
plt.figure(figsize=(8, 5))
for i in range(num_classes_minivgg_model):
    plt.plot(fpr_minivgg[i], tpr_minivgg[i], label=f'ROC curve of class {class_labels[i]} (area = {roc_auc_minivgg[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for MiniVGG Model')
plt.legend(loc="lower right")
plt.grid()
plt.savefig(os.path.join(minivgg_folder, 'roc_curve.png'))  # Save the plot
plt.show()


input("Press Enter to continue...")

# Train and evaluate the selected model
densenet_model = densenet_model(input_shape, num_classes) 
# Save the model architecture plot in densenet_folder
tf.keras.utils.plot_model(densenet_model, to_file=os.path.join(densenet_folder, 'densenet_model_architecture.png'), show_shapes=True)
densenet_model.summary()
history_densenet, predictions, true_classes, class_labels, fpr, tpr, roc_auc, cm = train_and_evaluation_densenet(densenet_model, train_generator, valid_generator, test_generator, epochs=100)  # Adjust epochs as needed

# Plot training vs validation loss
plt.figure(figsize=(8, 5))
plt.plot(history_densenet.history['loss'], label='Training Loss')
plt.plot(history_densenet.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss for DenseNet densenet_model')
plt.grid()
plt.savefig(os.path.join(densenet_folder, 'training_vs_validation_loss.png'))  # Save the plot
plt.show()

# Plot training vs validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history_densenet.history['accuracy'], label='Training Accuracy')
plt.plot(history_densenet.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy for DenseNet Model')
plt.grid()
plt.savefig(os.path.join(densenet_folder, 'training_vs_validation_accuracy.png'))  # Save the plot
plt.show()

# Plot confusion matrix
true_classes_densenet_model = test_generator.classes
predicted_classes_densenet_model = np.argmax(densenet_model.predict(test_generator), axis=1)
cm_densenet_model = confusion_matrix(true_classes_densenet_model, predicted_classes_densenet_model)
plt.figure(figsize=(5, 5))
sns.heatmap(cm_densenet_model, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for DenseNet densenet_model')
plt.savefig(os.path.join(densenet_folder, 'confusion_matrix.png'))  # Save the plot
plt.show()

input("Press Enter to continue...")

# Extract values for ROC curve plotting
num_classes_densenet_model = len(class_labels)

# Plot ROC Curve
plt.figure(figsize=(8, 8))
for i in range(num_classes_densenet_model):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for DenseNet Model')
plt.legend(loc="lower right")
plt.savefig(os.path.join(densenet_folder, 'roc_curve.png'))  # Save the plot
plt.show()


input("Press Enter to continue...")

# Build ResNet model
resnet_model = resnet_model(input_shape, num_classes)
resnet_model.summary()

tf.keras.utils.plot_model(resnet_model, to_file=os.path.join(resnet_folder, 'resnet_model_architecture.png'), show_shapes=True) 

history_resnet_model, predictions_resnet_model, true_classes_resnet_model, class_labels_resnet_model, fpr_resnet_model, tpr_resnet_model, roc_auc_resnet_model, cm_resnet_model = train_and_evaluation_resnet(resnet_model, train_generator, valid_generator,  test_generator, epochs=40) # change it back to 40 after testing

# Plot training vs validation loss
plt.figure(figsize=(8, 5))
plt.plot(history_resnet_model.history['loss'], label='Training Loss')
plt.plot(history_resnet_model.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss for ResNet Model')
plt.grid()
plt.savefig(os.path.join(resnet_folder, 'training_vs_validation_loss.png'))  # Save the plot
plt.show()

# Plot training vs validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history_resnet_model.history['accuracy'], label='Training Accuracy')
plt.plot(history_resnet_model.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy for ResNet Model')
plt.grid()
plt.savefig(os.path.join(resnet_folder, 'training_vs_validation_accuracy.png'))  # Save the plot
plt.show()

# Plot confusion matrix
true_classes_resnet_model = test_generator.classes
predicted_classes_resnet_model = np.argmax(resnet_model.predict(test_generator), axis=1)
cm_resnet_model = confusion_matrix(true_classes_resnet_model, predicted_classes_resnet_model)
plt.figure(figsize=(5, 5))
sns.heatmap(cm_resnet_model, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for ResNet Model')
plt.savefig(os.path.join(resnet_folder, 'confusion_matrix.png'))  # Save the plot
plt.show()

input("Press Enter to continue...")

# Extract values for ROC curve plotting
num_classes_resnet_model = len(class_labels_resnet_model)

# Plot ROC Curve
plt.figure(figsize=(8, 8))
for i in range(num_classes_resnet_model):
    plt.plot(fpr_resnet_model[i], tpr_resnet_model[i], label=f'ROC curve of class {class_labels_resnet_model[i]} (area = {roc_auc_resnet_model[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for ResNet Model')
plt.legend(loc="lower right")
plt.savefig(os.path.join(resnet_folder, 'roc_curve.png'))  # Save the plot
plt.show()

input("Press Enter to continue...")

# Train and evaluate MiniVGG_var1 model
minivgg_var1_model = miniVGG_model_Var1(input_shape, num_classes)
# Save the model architecture plot in minivgg_var1_folder
tf.keras.utils.plot_model(minivgg_var1_model, to_file=os.path.join(minivgg_var1_folder, 'minivgg_var1_model_architecture.png'), show_shapes=True)
minivgg_var1_model.summary()
history_minivgg_var1_model, predictions_minivgg_var1_model, true_classes_minivgg_var1_model, class_labels_minivgg_var1_model, fpr_minivgg_var1_model, tpr_minivgg_var1_model, roc_auc_minivgg_var1_model, cm_minivgg_var1_model = train_and_evaluate_miniVGG_Var1(minivgg_var1_model, train_generator, valid_generator, test_generator, epochs=100)  # Adjust epochs as needed

# Plot training vs validation loss
plt.figure(figsize=(8, 5))
plt.plot(history_minivgg_var1_model.history['loss'], label='Training Loss')
plt.plot(history_minivgg_var1_model.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss for MiniVGG_var1 Model')
plt.grid()
plt.savefig(os.path.join(minivgg_var1_folder, 'training_vs_validation_loss.png'))  # Save the plot
plt.show()

# Plot training vs validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history_minivgg_var1_model.history['accuracy'], label='Training Accuracy')
plt.plot(history_minivgg_var1_model.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy for MiniVGG_var1 Model')
plt.grid()
plt.savefig(os.path.join(minivgg_var1_folder, 'training_vs_validation_accuracy.png'))  # Save the plot
plt.show()

input("Press Enter to continue...")

# Plot confusion matrix
true_classes_minivgg_var1_model = test_generator.classes
predicted_classes_minivgg_var1_model = np.argmax(minivgg_var1_model.predict(test_generator), axis=1)
cm_minivgg_var1_model = confusion_matrix(true_classes_minivgg_var1_model, predicted_classes_minivgg_var1_model)
plt.figure(figsize=(5, 5))
sns.heatmap(cm_minivgg_var1_model, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for MiniVGG_var1 Model')
plt.savefig(os.path.join(minivgg_var1_folder, 'confusion_matrix.png'))  # Save the plot
plt.show()


# Extract values for ROC curve plotting
num_classes_minivgg_var1_model = len(class_labels_minivgg_var1_model)

# Plot ROC Curve
plt.figure(figsize=(8, 8))
for i in range(num_classes_minivgg_var1_model):
    plt.plot(fpr_minivgg_var1_model[i], tpr_minivgg_var1_model[i], label=f'ROC curve of class {class_labels_minivgg_var1_model[i]} (area = {roc_auc_minivgg_var1_model[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for MiniVGG_var1 Model')
plt.legend(loc="lower right")
plt.savefig(os.path.join(minivgg_var1_folder, 'roc_curve.png'))  # Save the plot
plt.show()

input("Press Enter to continue...")

# Create an instance of the DenseNetModelVar
densenet_model_var = densenet_model_var(input_shape, num_classes)

# Save the model architecture plot in densenet_var_folder
tf.keras.utils.plot_model(densenet_model_var, to_file=os.path.join(densenet_var_folder, 'densenet_var_model_architecture.png'), show_shapes=True)
densenet_model_var.summary()

# Train and evaluate the model
history_densenet_var, predictions_densenet_var, true_classes_densenet_var, class_labels_densenet_var, fpr_densenet_var, tpr_densenet_var, roc_auc_densenet_var, cm_densenet_var = train_and_evaluate_densenet_var(densenet_model_var, train_generator, valid_generator, test_generator, epochs=100)  # Adjust epochs as needed 100

# Plot training vs validation loss
plt.figure(figsize=(8, 5))
plt.plot(history_densenet_var.history['loss'], label='Training Loss')
plt.plot(history_densenet_var.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss for DenseNet densenet_model_var')
plt.grid()
plt.savefig(os.path.join(densenet_var_folder, 'training_vs_validation_loss.png'))  # Save the plot
plt.show()

# Plot training vs validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history_densenet_var.history['accuracy'], label='Training Accuracy')
plt.plot(history_densenet_var.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy for DenseNet Model')
plt.grid()
plt.savefig(os.path.join(densenet_var_folder, 'training_vs_validation_accuracy.png'))  # Save the plot
plt.show()

input("Press Enter to continue...")

# Plot confusion matrix
cm_densenet_var = confusion_matrix(true_classes_densenet_var, np.argmax(predictions_densenet_var, axis=1))
plt.figure(figsize=(5, 5))
sns.heatmap(cm_densenet_var, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for DenseNet densenet_model_var')
plt.savefig(os.path.join(densenet_var_folder, 'confusion_matrix.png'))  # Save the plot
plt.show()

