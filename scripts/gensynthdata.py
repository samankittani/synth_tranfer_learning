import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing.image import array_to_img


def gen_synth_data():
    # Only rescaling for validation
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        'datasets/chest_xray/test',  # Path to validation data
        # Resize images to 150x150 for the model
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary',              # For binary classification
        color_mode='grayscale',
        shuffle=False
    )

    model = load_model('scripts/base_model.h5')

    right_dir = 'datasets/synthdata/RIGHT/'
    wrong_dir = 'datasets/synthdata/WRONG/'

    # Create directories if they don't exist
    os.makedirs(right_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    # Iterate over the test generator
    for i in range(len(test_generator)):
        # Get the next batch of images and labels
        test_images, test_labels = test_generator.next()

        # Make predictions on this batch
        predicted_labels = model.predict(test_images)
        predicted_labels = np.round(predicted_labels).astype(int).flatten()

        # Save images to the correct directory based on prediction accuracy
        for j, (image, predicted_label, true_label) in enumerate(
            zip(test_images,
                predicted_labels,
                test_labels)):

            # Convert image array back to image
            img = array_to_img(image)

            # Construct the filename (assuming you might
            # want to avoid overwriting and include batch and index info)
            filename = f'img_{i:0{4}}_{j:0{4}}_{round(true_label)}.png'

            # Check if prediction is correct and choose the directory
            if predicted_label == true_label:
                img.save(os.path.join(right_dir, filename))
            else:
                img.save(os.path.join(wrong_dir, filename))
