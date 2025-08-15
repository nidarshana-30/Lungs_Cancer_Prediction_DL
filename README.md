# 🫁 Lung Cancer Prediction using Deep Learning


📌 Project Overview

This project uses deep learning to classify CT scan images as cancerous or non-cancerous, assisting radiologists in early and accurate lung cancer detection.

Objective: Detect lung cancer from CT scan images.

Model Used: Pre-trained VGG16 CNN architecture (fine-tuned).

Dataset: LIDC-IDRI or custom-labeled CT scan images.

Framework: TensorFlow / Keras

📁 Dataset Description

Classes:

0: Non-cancerous

1: Cancerous

Preprocessing Steps:

Resized images to 224x224 pixels

Normalized pixel values

Optional data augmentation (rotation, flipping, zooming) for improved performance

Dataset Source:

LIDC-IDRI dataset on The Cancer Imaging Archive

⚠️ Note: Dataset not included due to size; users should download and place it in a dataset/ folder.

🧠 Model Architecture

The model uses a fine-tuned VGG16 architecture:

Base Model: VGG16 (pre-trained on ImageNet, top layers removed)

Added Layers:

GlobalAveragePooling2D

Dense layer with ReLU activation

Dropout for regularization

Dense output layer with Sigmoid activation for binary classification

Loss Function: Binary Crossentropy

Optimizer: Adam

Metrics: Accuracy, Precision, Recall

🔧 Implementation Details

Load and preprocess images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
val_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


Build the model

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)


Compile and train

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=val_generator, epochs=25)


Evaluate model

model.evaluate(val_generator)


Predict on new images

from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('path_to_ct_image.jpg', target_size=(224,224))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
print("Cancerous" if prediction[0][0]>0.5 else "Non-cancerous")

📈 Results

Training Accuracy: ~95% (example)

Validation Accuracy: ~92% (example)

Confusion matrix and ROC-AUC curve can be plotted to visualize performance.

📝 Future Enhancements

* Use other architectures like ResNet50, DenseNet121 for comparison

* Apply transfer learning with fine-tuning deeper layers

* Deploy as a web app or desktop GUI for real-time prediction

* Integrate with medical image DICOM preprocessing pipelines
