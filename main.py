import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import GlobalMaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import EarlyStopping


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow will run on GPU")
    else:
        print("TensorFlow will run on CPU")

    # Define ImageDataGenerators for training
    # and validation (with augmentation for training)
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values to [0, 1]
        rotation_range=20,       # Random rotations
        width_shift_range=0.2,   # Random horizontal shifts
        height_shift_range=0.2,  # Random vertical shifts
        shear_range=0.2,         # Shear transformations
        zoom_range=0.2,          # Random zooms
        horizontal_flip=True
    )

    # Define ImageDataGenerators for training
    # and validation (with augmentation for training)
    trans_train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values to [0, 1]
        rotation_range=20,       # Random rotations
        width_shift_range=0.2,   # Random horizontal shifts
        height_shift_range=0.2,  # Random vertical shifts
        shear_range=0.2,         # Shear transformations
        zoom_range=0.2,          # Random zooms
        horizontal_flip=True
    )

    # Only rescaling for validation
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Assuming the images are stored in directories with one directory per class
    train_generator = train_datagen.flow_from_directory(
        'datasets/synthdata/',  # Path to training data
        target_size=(150, 150),     # Resize images to 150x150 for the model
        batch_size=16,
        color_mode='grayscale',
        class_mode='binary'         # For binary classification
    )

    trans_train_generator = trans_train_datagen.flow_from_directory(
        'datasets/maindata/train/',  # Path to training data
        target_size=(150, 150),     # Resize images to 150x150 for the model
        batch_size=16,
        color_mode='grayscale',
        class_mode='binary'         # For binary classification
    )

    test_generator = test_datagen.flow_from_directory(
        'datasets/maindata/test/',  # Path to validation data
        # Resize images to 150x150 for the model
        target_size=(150, 150),
        batch_size=16,
        color_mode='grayscale',
        class_mode='binary'              # For binary classification
    )
    # Load pre-trained ResNet50 model
    # base_model = ResNet50(weights='imagenet', include_top=False,
    #                       input_shape=(150, 150, 1))

    # # Freeze the convolutional base
    # for layer in base_model.layers:
    #     layer.trainable = False

    # Create new model on top
    # x = GlobalAveragePooling2D()(base_model.output)
    # x = Dense(1024, activation='relu')(x)
    # predictions = Dense(1, activation='sigmoid')(x)
    # model = Model(inputs=base_model.input, outputs=predictions)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides=1, padding='same',
              activation='relu', input_shape=(150, 150, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    # Assume 'train_generator' and 'validation_generator' are prepared
    model.fit(train_generator, epochs=10)

    # Fine-tuning
    for layer in model.layers[:5]:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.00001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(trans_train_generator, epochs=10)

    # OUTPUTING THE RESULTS
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Predict classes with model
    predictions = model.predict(test_generator)
    # Assuming binary classification
    predictions = np.round(predictions).astype(int)

    # True labels
    true_classes = test_generator.classes

    # Generate confusion matrix
    cm = confusion_matrix(true_classes, predictions)
    print(cm)

    # Generate classification report
    print(classification_report(true_classes, predictions,
          target_names=test_generator.class_indices.keys()))

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(true_classes, predictions)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('plots/main.png')
    plt.show()

    model.save('scripts/transfer_model.h5')


main()
