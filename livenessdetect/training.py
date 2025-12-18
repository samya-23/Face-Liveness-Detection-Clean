# training.py
import matplotlib
matplotlib.use("Agg") # Fix for running on servers/headless environments

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import random

# Import the model from your file
from model import MiniVGG

# 1. Parse Command Line Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True, help="path to output model")
args = vars(ap.parse_args())

# 2. Config
INIT_LR = 1e-3
BS = 32
EPOCHS = 20 # Increased slightly for better results
img_height = 128
img_width = 128
num_classes = 2

# 3. Load Images
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))

if len(imagePaths) == 0:
    print(f"❌ Error: No images found in directory '{args['dataset']}'")
    print("   Make sure your folder structure is: dataset/real and dataset/fake")
    exit()

print(f"[INFO] Found {len(imagePaths)} images. Processing...")

data = []
labels = []
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    try:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (img_height, img_width))
        image = img_to_array(image)
        data.append(image)

        # Extract label from folder name (dataset/real/image.jpg -> real)
        label_name = imagePath.split(os.path.sep)[-2]
        
        # 1 = Fake, 0 = Real
        label = 1 if label_name == "fake" else 0
        labels.append(label)
    except Exception as e:
        print(f"Skipping bad image: {imagePath}")

# Normalize data
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split Data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# One-hot encoding
trainY = to_categorical(trainY, num_classes)
testY = to_categorical(testY, num_classes)

# Augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# 4. Compile Model
print("[INFO] compiling model...")
model = MiniVGG(width=img_width, height=img_height, depth=3, classes=num_classes)
opt = Adam(learning_rate=INIT_LR) # 'lr' is deprecated in TF2, use 'learning_rate'
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# 5. Train
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS
)

# 6. Evaluate
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["Real", "Fake"]))

# 7. Save Model
print(f"[INFO] serializing network to '{args['model']}'...")
model.save(args["model"], save_format="h5")
print("✅ Done! Model saved.")