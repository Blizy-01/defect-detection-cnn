import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURATION ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
DATA_DIR = '../data/raw'
MODEL_PATH = '../saved_models/defect_classifier.h5'

# --- STEP 1: CHECK MODEL ---
if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found!")
    print(f"Please run 'src/train.py' first to generate {MODEL_PATH}")
    exit()

# --- STEP 2: LOAD MODEL ---
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- STEP 3: PREPARE TEST DATA ---
# Note: We only rescale here, we do NOT augment (rotate/flip) for evaluation
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False 
)

# --- STEP 4: EVALUATE ---
print("Running evaluation on validation set...")
loss, accuracy = model.evaluate(validation_generator)

print("-" * 30)
print(f"FINAL TEST ACCURACY: {accuracy * 100:.2f}%")
print("-" * 30)
