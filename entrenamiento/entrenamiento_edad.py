import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from modeLR import modeLR

# === Parámetros ===
img_size = (224, 224)
batch_size = 32
epochs = 10
preprocessed_dir = "data_edad"

# === Generador de datos ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    preprocessed_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    preprocessed_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
assert len(class_names) == 4, f"Se esperaban 4 clases, pero se encontraron: {class_names}"
print("Clases:", class_names)

# === Crear modelo ===
modelo = modeLR(input_shape=(224, 224, 3), num_classes=len(class_names))

# === Callbacks ===
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
# === Entrenamiento ===
history = modelo.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stopping]
)

# === Guardar modelo e historial ===
modelo.save("programa/modelos/modelo_edad.h5")

# === Evaluación ===
val_loss, val_acc = modelo.evaluate(val_generator)
print(f"\nPrecision final en validacion: {val_acc:.2%}")

# === Matriz de confusión ===
val_generator.reset()
y_true, y_pred = [], []

for _ in range(len(val_generator)):
    X_batch, y_batch = next(val_generator)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(modelo.predict(X_batch), axis=1))

conf_matrix = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title("Matriz de Confusion (Absoluta)")

conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_matrix_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title("Matriz de Confusion (Porcentaje)")

plt.tight_layout()
plt.show()

# === Reporte ===
print("\nReporte de Clasificacion:\n", classification_report(y_true, y_pred, target_names=class_names))
