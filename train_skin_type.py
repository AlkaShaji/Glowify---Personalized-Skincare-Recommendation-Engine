import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# Paths
train_dir = r'C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\dataset\blossom_dataset\train'
valid_dir = r'C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\dataset\blossom_dataset\valid'
test_dir = r'C:\Users\ALKA\OneDrive\Desktop\ULTS\glowify\dataset\blossom_dataset\test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

target_size = (240, 240)  # EfficientNetB1 prefers this

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=target_size, batch_size=32, class_mode='categorical'
)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir, target_size=target_size, batch_size=32, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=target_size, batch_size=32, class_mode='categorical'
)

# Compute class weights
class_labels = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_labels), y=class_labels)
class_weight_dict = dict(enumerate(class_weights))

# Load EfficientNetB1
base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
base_model.trainable = False

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',  # label_smoothing removed
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3, min_lr=1e-6)

# Initial Training
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, lr_scheduler]
)

# Unfreeze all layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers:
    layer.trainable = True

# Recompile with lower LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning
history_finetune = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save model
model.save("efficientnetb1_skin_type_model.keras")

# Plot training curves
def plot_history(histories, titles):
    for history, title in zip(histories, titles):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label='Train Accuracy')
        plt.plot(epochs, val_acc, label='Val Accuracy')
        plt.title(f'{title} - Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Val Loss')
        plt.title(f'{title} - Loss')
        plt.legend()

        plt.show()

plot_history([history, history_finetune], ['Initial Training', 'Fine-Tuning'])
