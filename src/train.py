import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- CONFIGURATION ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = '../data/raw'
MODEL_SAVE_PATH = '../saved_models/defect_classifier.h5'

# Ensure the save directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# --- STEP 1: DATA LOADING & AUGMENTATION ---
print("Initializing Data Generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# --- STEP 2: BUILD MODEL ---
print("Building CNN Model...")
model = Sequential([
    # Layer 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    
    # Layer 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Layer 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Classification Head
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- STEP 3: TRAIN ---
print(f"Starting training for {EPOCHS} epochs...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- STEP 4: SAVE ---
model.save(MODEL_SAVE_PATH)
print(f"Success! Model saved to {MODEL_SAVE_PATH}")
