# ============================================
# INKVISION AI
# Handwritten Digit Recognition System
# CNN Model Training
# ============================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ============================================
# 1. CREATE REQUIRED DIRECTORIES
# ============================================

os.makedirs("model", exist_ok=True)
os.makedirs("assets", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("\n====================================")
print("       INKVISION AI")
print(" CNN MODEL TRAINING STARTED")
print("====================================\n")


# ============================================
# 2. LOAD MNIST DATASET
# ============================================

print("Loading MNIST Dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Dataset Loaded Successfully!")

print("Training Images:", x_train.shape)
print("Testing Images:", x_test.shape)


# ============================================
# 3. DATA PREPROCESSING
# ============================================

print("\nPreprocessing Images...")

# Normalize pixel values from 0-255 to 0-1

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images for CNN

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Preprocessing Completed!")

# ============================================
# 4. DISPLAY SAMPLE IMAGES
# ============================================

plt.figure(figsize=(10, 4))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")

    plt.title(f"Digit: {y_train[i]}")

    plt.axis("off")

plt.tight_layout()

plt.savefig("assets/sample_digits.png")

plt.show()


# ============================================
# 5. BUILD CNN MODEL
# ============================================

print("\nBuilding CNN Architecture...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    # First Convolution Layer

    tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Second Convolution Layer

    tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Flatten Layer

    tf.keras.layers.Flatten(),

    # Fully Connected Layer

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout to reduce overfitting

    tf.keras.layers.Dropout(0.3),

    # Output Layer (0-9)

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )

])

# ============================================
# 6. COMPILE MODEL
# ============================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nCNN Model Summary:")

model.summary()


# ============================================
# 7. TRAIN CNN MODEL
# ============================================

print("\nStarting Model Training...\n")

history = model.fit(

    x_train,
    y_train,

    epochs=5,

    batch_size=64,

    validation_split=0.1,

    verbose=1

)


# ============================================
# 8. EVALUATE MODEL
# ============================================

print("\nEvaluating Model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\n====================================")

print("MODEL EVALUATION RESULTS")

print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

print("Test Loss:", round(test_loss, 4))

print("====================================")


# ============================================
# 9. SAVE TRAINED MODEL
# ============================================

model.save("model/digit_cnn.keras")

print("\nModel Saved Successfully!")

# ============================================
# 10. TRAINING ACCURACY GRAPH
# ============================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Model Accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("reports/accuracy_graph.png")

plt.show()


# ============================================
# 11. TRAINING LOSS GRAPH
# ============================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Model Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("reports/loss_graph.png")

plt.show()


# ============================================
# 12. CONFUSION MATRIX
# ============================================

print("\nGenerating Confusion Matrix...")

y_pred = model.predict(x_test, verbose=0)

y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=range(10),

    yticklabels=range(10)

)

plt.title("Handwritten Digit Recognition - Confusion Matrix")

plt.xlabel("Predicted Digit")

plt.ylabel("Actual Digit")

plt.savefig("reports/confusion_matrix.png")

plt.show()


# ============================================
# 13. CLASSIFICATION REPORT
# ============================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_classes
    )
)


# ============================================
# 14. FINAL MESSAGE
# ============================================

print("\n====================================")

print(" INKVISION AI MODEL TRAINING COMPLETE!")

print(" Model: CNN")

print(" Dataset: MNIST")

print(" Classes: 0-9")

print("====================================")

print("\nGenerated Files:")

print("1. model/digit_cnn.keras")

print("2. assets/sample_digits.png")

print("3. reports/accuracy_graph.png")

print("4. reports/loss_graph.png")

print("5. reports/confusion_matrix.png")

print("\nReady for the next development phase!")