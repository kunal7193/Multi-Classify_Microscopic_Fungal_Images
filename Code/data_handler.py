# data_handler.py

from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import shutil
import cv2
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

class DataHandler:
    def __init__(self, main_folder, subfolders, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.main_folder = main_folder
        self.subfolders = subfolders
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.img_width = 128  # Set your desired image width
        self.img_height = 128  # Set your desired image height

    def create_directories(self):
        for directory in ['train', 'valid', 'test']:
            dir_path = os.path.join(self.main_folder, directory)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                for subfolder in self.subfolders:
                    subfolder_path = os.path.join(dir_path, subfolder)
                    os.makedirs(subfolder_path, exist_ok=True)

    def is_image_file(self, file_path):
        # Exclude ".DS_Store" files
        if file_path.endswith(".DS_Store"):
            return False

        try:
            # Attempt to open the file using Pillow
            with Image.open(file_path):
                return True
        except (IOError, OSError):
            # The file is not a valid image file
            return False

    def split_data(self):
        for subfolder in self.subfolders:
            files = os.listdir(os.path.join(self.main_folder, subfolder))
            random.shuffle(files)

            train_split = int(self.train_ratio * len(files))
            val_split = int(self.val_ratio * len(files))

            train_files = files[:train_split]
            val_files = files[train_split:train_split + val_split]
            test_files = files[train_split + val_split:]

            for directory, file_list in [('train', train_files), ('valid', val_files), ('test', test_files)]:
                for file in file_list:
                    file_path = os.path.join(self.main_folder, subfolder, file)
                    dest_path = os.path.join(self.main_folder, directory, subfolder, file)
                    if self.is_image_file(file_path) and not os.path.exists(dest_path):
                        shutil.move(file_path, dest_path)

    def load_and_split_data(self):
        self.create_directories()
        self.split_data()
    
    def count_images(self, directory):
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len(files)
        return count

    def print_image_counts(self):
        print("Number of Images in Train Directory:", self.count_images(os.path.join(self.main_folder, 'train')))
        print("Number of Images in Validation Directory:", self.count_images(os.path.join(self.main_folder, 'valid')))
        print("Number of Images in Test Directory:", self.count_images(os.path.join(self.main_folder, 'test')))

    def count_images_in_class(self, dataset, class_folder):
        class_path = os.path.join(dataset, class_folder)
        return len(os.listdir(class_path))

    def print_image_counts_by_class(self):
        for dataset in ['train', 'valid', 'test']:
            print(f"\nNumber of Images in Each Class in {dataset.capitalize()} Dataset:")
            for class_folder in self.subfolders:
                num_images = self.count_images_in_class(os.path.join(self.main_folder, dataset), class_folder)
                print(f"Class {class_folder}: {num_images} images")
    
    def display_images(self, dataset_directory):
        fig, axs = plt.subplots(1, len(self.subfolders), figsize=(15, 3))  # Adjusted to create subplots based on the number of classes
        axs = axs.ravel()
        for i, class_folder in enumerate(self.subfolders):
            img_path = os.path.join(dataset_directory, class_folder, os.listdir(os.path.join(dataset_directory, class_folder))[0])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i].imshow(img)
            axs[i].set_title(f"Class: {class_folder}")
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()

    def create_datagens(self):
        img_width, img_height = 128, 128

        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1.0/255)

        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.main_folder, 'train'),
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical'
        )

        valid_generator = test_datagen.flow_from_directory(
            os.path.join(self.main_folder, 'valid'),
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.main_folder, 'test'),
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, valid_generator, test_generator
    
    def get_train_image_path(self, class_name):
        train_class_folder = os.path.join(self.main_folder, 'train', class_name)
        train_images = os.listdir(train_class_folder)
        random_test_image = random.choice(train_images)
        return os.path.join(train_class_folder, random_test_image)


