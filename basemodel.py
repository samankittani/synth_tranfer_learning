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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow will run on GPU")
else:
    print("TensorFlow will run on CPU")


# Define ImageDataGenerators for training
# and validation (with augmentation for training)
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    rotation_range=20,       # Random rotations
    width_shift_range=0.2,   # Random horizontal shifting
    height_shift_range=0.2,  # Random vertical shifting
    horizontal_flip=True     # Horizontal flipping
)

# Only rescaling for validation
test_datagen = ImageDataGenerator(rescale=1./255)

# Only rescaling for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Assuming the images are stored in directories with one directory per class
train_generator = train_datagen.flow_from_directory(
    './chest_xray/train',  # Path to training data
    target_size=(224, 224),     # Resize images to 224x224 for the model
    batch_size=16,
    class_mode='binary'         # For binary classification
)

test_generator = test_datagen.flow_from_directory(
    './chest_xray/test',  # Path to validation data
    target_size=(224, 224),          # Resize images to 224x224 for the model
    batch_size=16,
    class_mode='binary'              # For binary classification
)

validation_generator = val_datagen.flow_from_directory(
    './chest_xray/val',  # Path to validation data
    target_size=(224, 224),          # Resize images to 224x224 for the model
    batch_size=16,
    class_mode='binary'              # For binary classification
)


# Load pre-trained ResNet50 model
# base_model = ResNet50(weights='imagenet', include_top=False,
#                       input_shape=(224, 224, 3))

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
          activation='relu', input_shape=(224, 224, 3)))
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


early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
# Assume 'train_generator' and 'validation_generator' are prepared
model.fit(train_generator, epochs=20,
          validation_data=validation_generator, callbacks=[early_stopping])

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
plt.savefig('plot1.png')
plt.show()

# Take a batch of images from the test generator
test_images, test_labels = next(test_generator)

# Predicting from the model
predicted_labels = model.predict(test_images)

# Show the images and the model predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i])
    plt.title(f"Actual: {test_labels[i]}, Predicted: {predicted_labels[i]}")
    plt.axis("off")
plt.savefig('plot2.png')

model.save('base_model.h5')
