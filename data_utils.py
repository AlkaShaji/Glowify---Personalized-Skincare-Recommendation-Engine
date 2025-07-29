import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def multilabel_generator(image_paths, labels, batch_size=32, target_size=(224, 224), augment=True):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=augment,
        rotation_range=10 if augment else 0,
        zoom_range=0.1 if augment else 0,
        width_shift_range=0.1 if augment else 0,
        height_shift_range=0.1 if augment else 0
    )

    num_samples = len(image_paths)
    
    while True:
        idxs = np.arange(num_samples)
        np.random.shuffle(idxs)

        for i in range(0, num_samples, batch_size):
            batch_ids = idxs[i:i+batch_size]
            batch_images = []
            batch_labels = []

            for j in batch_ids:
                try:
                    img = load_img(image_paths[j], target_size=target_size)
                    img = img_to_array(img)
                    batch_images.append(img)
                    batch_labels.append(labels[j])
                except Exception as e:
                    print(f"[WARNING] Skipping file {image_paths[j]}: {e}")
                    continue  # skip problematic file

            if not batch_images:
                continue  # skip if no valid images in batch

            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)

            batch_images = datagen.standardize(batch_images)
            yield batch_images, batch_labels
