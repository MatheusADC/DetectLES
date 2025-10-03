import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
from datetime import datetime
import shutil

# Configurações
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 2
CLASSES = ['lupus', 'saudavel']

# Caminhos
DATASET_PATH = 'dataset'
MODEL_OUTPUT_PATH = 'modelo-lupus2'

print("=== TREINAMENTO CNN COM PYTHON ===")
print(f"Tamanho da imagem: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Épocas: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Classes: {CLASSES}")

# Criar pasta de saída
if os.path.exists(MODEL_OUTPUT_PATH):
    shutil.rmtree(MODEL_OUTPUT_PATH)
os.makedirs(MODEL_OUTPUT_PATH)

# Carregar dataset
print("\n=== CARREGANDO DATASET ===")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

print(f"Dataset de treino: {len(train_ds)} batches")
print(f"Dataset de validação: {len(val_ds)} batches")

# Normalizar dados
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Criar modelo CNN
print("\n=== CRIANDO MODELO CNN ===")
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compilar modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Arquitetura do modelo:")
model.summary()

# Treinar modelo
print("\n=== INICIANDO TREINAMENTO ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# Avaliar modelo
print("\n=== AVALIANDO MODELO ===")
test_loss, test_accuracy = model.evaluate(val_ds, verbose=0)
print(f"Accuracy no teste: {test_accuracy:.4f}")
print(f"Loss no teste: {test_loss:.4f}")

# Gerar predições para matriz de confusão
print("\n=== GERANDO MATRIZ DE CONFUSÃO ===")
y_pred = []
y_true = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
print("\nMatriz de Confusão:")
print("Predição\\Real", CLASSES)
for i, class_name in enumerate(CLASSES):
    row = f"{class_name}\t"
    for j in range(len(CLASSES)):
        row += f"{cm[i][j]}\t"
    print(row)

# Relatório de classificação
print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Calcular métricas
precision_lupus = report['lupus']['precision']
recall_lupus = report['lupus']['recall']
f1_lupus = report['lupus']['f1-score']

precision_saudavel = report['saudavel']['precision']
recall_saudavel = report['saudavel']['recall']
f1_saudavel = report['saudavel']['f1-score']

print(f"\n=== MÉTRICAS DETALHADAS ===")
print(f"Lupus:")
print(f"  Precisão: {precision_lupus:.4f}")
print(f"  Recall: {recall_lupus:.4f}")
print(f"  F1-Score: {f1_lupus:.4f}")
print(f"Saudável:")
print(f"  Precisão: {precision_saudavel:.4f}")
print(f"  Recall: {recall_saudavel:.4f}")
print(f"  F1-Score: {f1_saudavel:.4f}")

# Gerar gráficos
print("\n=== GERANDO GRÁFICOS ===")

# Gráfico 1: Matriz de Confusão
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Matriz de Confusão - CNN Lupus vs Saudável')
plt.ylabel('Classe Real')
plt.xlabel('Classe Predita')
plt.tight_layout()
plt.savefig(f'{MODEL_OUTPUT_PATH}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 2: Histórico de Treinamento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Accuracy durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f'{MODEL_OUTPUT_PATH}/training_history.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 3: Métricas por Classe
plt.figure(figsize=(10, 6))
metrics = ['Precisão', 'Recall', 'F1-Score']
lupus_scores = [precision_lupus, recall_lupus, f1_lupus]
saudavel_scores = [precision_saudavel, recall_saudavel, f1_saudavel]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, lupus_scores, width, label='Lupus', color='#3498DB')
plt.bar(x + width/2, saudavel_scores, width, label='Saudável', color='#27AE60')

plt.xlabel('Métricas')
plt.ylabel('Score')
plt.title('Métricas por Classe')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)

# Adicionar valores nas barras
for i, (lupus, saudavel) in enumerate(zip(lupus_scores, saudavel_scores)):
    plt.text(i - width/2, lupus + 0.01, f'{lupus:.3f}', ha='center', va='bottom')
    plt.text(i + width/2, saudavel + 0.01, f'{saudavel:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{MODEL_OUTPUT_PATH}/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 4: Distribuição das Predições
plt.figure(figsize=(8, 6))
pred_counts = np.bincount(y_pred)
true_counts = np.bincount(y_true)

x = np.arange(len(CLASSES))
width = 0.35

plt.bar(x - width/2, true_counts, width, label='Real', alpha=0.7)
plt.bar(x + width/2, pred_counts, width, label='Predito', alpha=0.7)

plt.xlabel('Classes')
plt.ylabel('Quantidade')
plt.title('Distribuição das Classes')
plt.xticks(x, CLASSES)
plt.legend()

plt.tight_layout()
plt.savefig(f'{MODEL_OUTPUT_PATH}/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("Gráficos salvos:")
print("- confusion_matrix.png")
print("- training_history.png") 
print("- metrics_comparison.png")
print("- class_distribution.png")

# Salvar modelo
print("\n=== SALVANDO MODELO ===")

# Salvar modelo TensorFlow
model.save(f'{MODEL_OUTPUT_PATH}/model_tf')

# Converter para TensorFlow.js
print("Convertendo para TensorFlow.js...")
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, f'{MODEL_OUTPUT_PATH}/model_tfjs')

# Criar metadata.json
metadata = {
    "tfjsVersion": "4.17.0",
    "tmVersion": "2.4.10", 
    "packageVersion": "0.8.4-alpha2",
    "packageName": "@teachablemachine/image",
    "timeStamp": datetime.now().isoformat(),
    "userMetadata": {},
    "modelName": "lupus-detection-model-trained",
    "labels": CLASSES,
    "imageSize": IMAGE_SIZE,
    "trainingMetrics": {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "precision_lupus": float(precision_lupus),
        "recall_lupus": float(recall_lupus),
        "f1_lupus": float(f1_lupus),
        "precision_saudavel": float(precision_saudavel),
        "recall_saudavel": float(recall_saudavel),
        "f1_saudavel": float(f1_saudavel)
    }
}

with open(f'{MODEL_OUTPUT_PATH}/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# Copiar arquivos do TensorFlow.js para a pasta principal
import shutil
import glob

# Encontrar arquivos do modelo TensorFlow.js
tfjs_files = glob.glob(f'{MODEL_OUTPUT_PATH}/model_tfjs/*')
for file_path in tfjs_files:
    filename = os.path.basename(file_path)
    shutil.copy2(file_path, f'{MODEL_OUTPUT_PATH}/{filename}')

print(f"Modelo salvo em: {MODEL_OUTPUT_PATH}")
print("Arquivos criados:")
print("- model.json")
print("- weights.bin") 
print("- metadata.json")
print("- model_tf/ (modelo TensorFlow)")
print("- model_tfjs/ (modelo TensorFlow.js)")

print("\n=== TREINAMENTO CONCLUÍDO COM SUCESSO! ===")
print(f"Accuracy final: {test_accuracy:.4f}")
print(f"Loss final: {test_loss:.4f}")
