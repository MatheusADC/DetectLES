import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score
import os
import json
from datetime import datetime
import shutil
import tensorflowjs as tfjs

# =====================================
# CONFIGURAÇÕES
# =====================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
NUM_CLASSES = 2
CLASSES = ['lupus', 'saudavel']

DATASET_PATH = "dataset2"
MODEL_OUTPUT_PATH = "modelo-lupus5"

# =====================================
# LIMPAR MODELO ANTIGO
# =====================================

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

# =====================================
# MODELO (MobileNetV2)
# =====================================

print("=== CRIANDO MODELO ===")

base_model = keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

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

# =====================================
# TREINAMENTO
# =====================================

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

# =====================================
# PREDIÇÕES E MATRIZ DE CONFUSÃO
# =====================================

print("\n=== GERANDO MATRIZ DE CONFUSÃO ===")

pred_probs = model.predict(val_ds)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_ds.classes

cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusão:\n", cm)

# =====================================
# SALVAR MATRIZ DE CONFUSÃO (PNG)
# =====================================

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, "matriz_confusao.png"))
plt.close()

print("Imagem 'matriz_confusao.png' criada!")

# =====================================
# MÉTRICAS: RECAL, PRECISÃO, F1, ESPECIFICIDADE
# =====================================

print("\n=== CALCULANDO MÉTRICAS ===")

precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

# Especificidade
specificity = []
for i in range(NUM_CLASSES):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity.append(tn / (tn + fp))

# =====================================
# GRÁFICO DE BARRAS DAS MÉTRICAS
# =====================================

x = np.arange(NUM_CLASSES)
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - width*1.5, precision, width, label="Precisão")
plt.bar(x - width*0.5, recall, width, label="Recall")
plt.bar(x + width*0.5, specificity, width, label="Especificidade")
plt.bar(x + width*1.5, f1, width, label="F1-Score")

plt.xticks(x, CLASSES)
plt.ylabel("Valor")
plt.ylim(0, 1)
plt.title("Métricas por Classe")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, "metricas_barras.png"))
plt.close()

print("Imagem 'metricas_barras.png' criada!")

# =====================================
# SALVANDO MODELO (TF, H5, TFJS)
# =====================================

print("\n=== SALVANDO MODELOS ===")

model_tf_path = os.path.join(MODEL_OUTPUT_PATH, "model_tf")
model_tfjs_path = os.path.join(MODEL_OUTPUT_PATH, "model_tfjs")

os.makedirs(model_tf_path, exist_ok=True)
os.makedirs(os.path.join(model_tf_path, "variables"), exist_ok=True)

print("Salvando modelo TensorFlow SavedModel...")
model.save(model_tf_path, save_format="tf")

print("Salvando modelo H5...")
model.save(os.path.join(MODEL_OUTPUT_PATH, "model.h5"))

print("Convertendo para TensorFlow.js...")
os.makedirs(model_tfjs_path, exist_ok=True)
tfjs.converters.save_keras_model(model, model_tfjs_path)

# =====================================
# METADATA
# =====================================

metadata = {
    "labels": CLASSES,
    "imageSize": IMAGE_SIZE,
    "timestamp": datetime.now().isoformat(),
    "validation_accuracy": float(test_acc),
    "validation_loss": float(test_loss)
}

with open(os.path.join(MODEL_OUTPUT_PATH, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n=== FINALIZADO COM SUCESSO ===")
