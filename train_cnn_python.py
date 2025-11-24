import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
from datetime import datetime
import shutil
import tensorflowjs as tfjs

# Configurações
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
NUM_CLASSES = 2
CLASSES = ['lupus', 'saudavel']

DATASET_PATH = "dataset2"
MODEL_OUTPUT_PATH = "modelo-lupus4"

# Apagar pasta antiga
if os.path.exists(MODEL_OUTPUT_PATH):
    shutil.rmtree(MODEL_OUTPUT_PATH)
os.makedirs(MODEL_OUTPUT_PATH)

print("=== CARREGANDO DATASET ===")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_ds = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_ds = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("=== CRIANDO MODELO (MobileNetV2 Transfer Learning) ===")

base_model = keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Congela pesos

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n=== TREINANDO MODELO ===")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

test_loss, test_acc = model.evaluate(val_ds, verbose=0)
print(f"Accuracy validação: {test_acc:.4f}")
print(f"Loss validação: {test_loss:.4f}")

pred_probs = model.predict(val_ds)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_ds.classes

cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusão:\n", cm)

# ================================
# SALVANDO O MODELO (CORRIGIDO)
# ================================

print("\n=== SALVANDO MODELOS ===")

model_tf_path = os.path.join(MODEL_OUTPUT_PATH, "model_tf")
model_tfjs_path = os.path.join(MODEL_OUTPUT_PATH, "model_tfjs")

# Correção obrigatória no Windows + TF 2.3
os.makedirs(model_tf_path, exist_ok=True)
os.makedirs(os.path.join(model_tf_path, "variables"), exist_ok=True)

print("Salvando modelo TensorFlow SavedModel...")
model.save(model_tf_path, save_format="tf")

print("Salvando modelo H5...")
model.save(os.path.join(MODEL_OUTPUT_PATH, "model.h5"))

print("Convertendo para TensorFlow.js...")
os.makedirs(model_tfjs_path, exist_ok=True)
tfjs.converters.save_keras_model(model, model_tfjs_path)

# METADATA
metadata = {
    "labels": CLASSES,
    "imageSize": IMAGE_SIZE,
    "timeStamp": datetime.now().isoformat(),
}
with open(os.path.join(MODEL_OUTPUT_PATH, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n=== FINALIZADO COM SUCESSO ===")
