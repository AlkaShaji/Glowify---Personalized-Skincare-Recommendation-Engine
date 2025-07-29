import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_utils import multilabel_generator
from sklearn.preprocessing import MultiLabelBinarizer
import json


# Parameters
IMG_SIZE = (224, 224)  # Resize images to this size
BATCH_SIZE = 32
EPOCHS = 20

# Set dataset paths
train_dir = r'C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\dataset\merged_skin_concern\train'
valid_dir = r'C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\dataset\merged_skin_concern\valid'

# Load image paths and labels
def load_image_paths_and_labels(folder_path):
    """
    Loads image paths and labels based on subfolders in the provided folder.
    Assumes that each subfolder represents a class.
    """
    image_paths = []
    labels = []
    class_names = os.listdir(folder_path)  # List subfolder names (class names)

    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):  # Ignore files, process subfolders only
            # Get all image files in the class folder
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                image_paths.append(img_path)
                labels.append([class_name])  # The class is the label

    return image_paths, labels

train_paths, train_labels = load_image_paths_and_labels(train_dir)
val_paths, val_labels = load_image_paths_and_labels(valid_dir)

# Multi-label binarization
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_labels)
y_val = mlb.transform(val_labels)
class_labels = mlb.classes_

# Save class labels
with open("concern_classes.json", "w") as f:
    json.dump(class_labels.tolist(), f)

# Print number of classes
NUM_CLASSES = len(class_labels)
print(f"Number of classes: {NUM_CLASSES}")

# Load ResNet50 (no top)
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
base_model.trainable = False  # We'll fine-tune later

# Custom head for multi-label classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-6)

# Training
model.fit(
    multilabel_generator(train_paths, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    validation_data=multilabel_generator(val_paths, y_val, batch_size=BATCH_SIZE),
    validation_steps=len(val_paths) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_schedule]
)

# Unfreeze and fine-tune last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    multilabel_generator(train_paths, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    validation_data=multilabel_generator(val_paths, y_val, batch_size=BATCH_SIZE),
    validation_steps=len(val_paths) // BATCH_SIZE,
    epochs=10,
    callbacks=[early_stop, lr_schedule]
)

# Save model
model.save("resnet_skin_concern_finetuned.keras")
