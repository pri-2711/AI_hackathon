"""
Vegetable Image Classifier — CNN from Scratch
==============================================
Dataset layout expected:
    DATASET_PATH/
        class_1/  (e.g. carrot/)
        class_2/
        class_3/
        class_4/
        class_5/

Update DATASET_PATH before running.
"""

# ─────────────────────────────────────────────
# CONFIGURATION  ← only line you need to edit
# ─────────────────────────────────────────────
DATASET_PATH = "balanced_dataset"    # <-- set this

IMAGE_SIZE    = (128, 128)   # resize all images to this
BATCH_SIZE    = 32
EPOCHS        = 20
NUM_CLASSES   = 5
RANDOM_SEED   = 42
MODEL_SAVE_AS = "vegetable_model.keras"
# ─────────────────────────────────────────────

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ── 1. Reproducibility ───────────────────────────────────────────────────────
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── 2. Data Generators ───────────────────────────────────────────────────────
# No augmentation — only rescaling and a 70 / 30 split.
datagen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    rotation_range   = 10,
    zoom_range       = 0.1,
    horizontal_flip  = True,
    brightness_range = [0.8, 1.2],
    validation_split = 0.30,        # 30 % held out (will become val + test)
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size  = IMAGE_SIZE,
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    subset       = "training",      # 70 %
    shuffle      = True,
    seed         = RANDOM_SEED,
)

# Full 30 % block — we'll manually split this into val (15 %) and test (15 %)
val_full_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size  = IMAGE_SIZE,
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    subset       = "validation",    # 30 %
    shuffle      = False,
    seed         = RANDOM_SEED,
)


# ── 3. Split the 30 % block into Validation (15 %) and Test (15 %) ───────────
def generator_to_arrays(gen):
    """Collect all batches from a generator into numpy arrays."""
    gen.reset()
    X_list, y_list = [], []
    steps = len(gen)
    for _ in range(steps):
        X_batch, y_batch = next(gen)
        X_list.append(X_batch)
        y_list.append(y_batch)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


print("\nLoading validation/test data into memory …")
X_val_full, y_val_full = generator_to_arrays(val_full_gen)

# Mid-point of the 30 % block → first 15 % = validation, last 15 % = test
mid = len(X_val_full) // 2
X_val,  y_val  = X_val_full[:mid],  y_val_full[:mid]
X_test, y_test = X_val_full[mid:],  y_val_full[mid:]

print(f"  Training   samples : {train_gen.samples}")
print(f"  Validation samples : {len(X_val)}")
print(f"  Test       samples : {len(X_test)}")


# ── 4. Class Indices ──────────────────────────────────────────────────────────
class_indices = train_gen.class_indices
print(f"\nClass indices : {class_indices}")


# ── 5. CNN Architecture ───────────────────────────────────────────────────────
def build_model(input_shape, num_classes):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        # Fully-connected head
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ], name="VegetableCNN")
    return model


model = build_model(
    input_shape = (*IMAGE_SIZE, 3),
    num_classes = NUM_CLASSES,
)
model.summary()


# ── 6. Compile ────────────────────────────────────────────────────────────────
model.compile(
    optimizer = Adam(),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)


# ── 7. Train ──────────────────────────────────────────────────────────────────
early = EarlyStopping(
    monitor             = 'val_loss',
    patience            = 5,
    restore_best_weights = True
)

print("\nTraining …\n")
history = model.fit(
    train_gen,
    epochs          = EPOCHS,
    validation_data = (X_val, y_val),
    callbacks       = [early],
    verbose         = 1,
)


# ── 8. Evaluate ───────────────────────────────────────────────────────────────
print("\n── Evaluation Results ──────────────────────────────────")

train_loss, train_acc = model.evaluate(train_gen, verbose=0)
val_loss,   val_acc   = model.evaluate(X_val,  y_val,  verbose=0)
test_loss,  test_acc  = model.evaluate(X_test, y_test, verbose=0)

print(f"  Training   — Loss: {train_loss:.4f}  |  Accuracy: {train_acc * 100:.2f}%")
print(f"  Validation — Loss: {val_loss:.4f}  |  Accuracy: {val_acc * 100:.2f}%")
print(f"  Test       — Loss: {test_loss:.4f}  |  Accuracy: {test_acc * 100:.2f}%")
print("────────────────────────────────────────────────────────\n")


# ── 9. Save ───────────────────────────────────────────────────────────────────
model.save(MODEL_SAVE_AS)
print(f"Model saved → {MODEL_SAVE_AS}")