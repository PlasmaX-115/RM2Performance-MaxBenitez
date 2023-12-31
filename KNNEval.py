#----------------------------------------------------------
#
# Date: 08-Sep-2023
#
#           A01752791 Maximiliano Benítez Ahumada
#----------------------------------------------------------

# Importar las bibliotecas necesarias
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap


# Cargar el conjunto de datos Digits como ejemplo
data = load_digits()
X, y = data.data, data.target

# Dividir los datos en tres conjuntos: entrenamiento, prueba y validación
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

best_accuracy = 0
best_k = 0

fig, axs = plt.subplots(2, 5, figsize=(15, 8))

# Crear listas para almacenar las precisiones en cada conjunto
precisions_train = []
precisions_test = []
precisions_validation = []

# Crear listas para almacenar el sesgo, la varianza y el nivel de ajuste
bias_scores = []
variance_scores = []
fit_scores = []

# Umbral para considerar una diferencia significativa:
umbral_diferencia = 0.02



# Probar diferentes valores de k y encontrar el mejor
for k in range(1, 11):  # Probar valores de k de 1 a 20
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Evaluar el modelo en el conjunto de prueba y validación

    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)  # Cambia X_test por tus datos de prueba reales
    y_pred_val = knn.predict(X_val)
    y_true_test = y_test  # Cambia y_test por tus etiquetas de prueba reales

    precision_train = accuracy_score(y_train, y_pred_train)
    precision_test = accuracy_score(y_test, y_pred_test)
    precision_validation = accuracy_score(y_val, y_pred_val)

    precisions_train.append(precision_train)
    precisions_test.append(precision_test)
    precisions_validation.append(precision_validation)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    # Calcular sesgo y varianza
    bias = 1 - accuracy_train
    variance = accuracy_train - accuracy_test

    # Calcular el nivel de ajuste
    if abs(bias) <= umbral_diferencia and abs(variance) <= umbral_diferencia:
        fit = "Adecuado"
    elif bias < 0 and variance > 0:
        fit = "Overfitting"
    else:
        fit = "Underfitting"

    bias_scores.append(bias)
    variance_scores.append(variance)
    fit_scores.append(fit)

# Generar el informe de clasificación
    report = classification_report(y_true_test, y_pred_test)
    cm = confusion_matrix(y_true_test,y_pred_test)
    num_classes = len(np.unique(y_test))
    class_labels = [str(i) for i in range(num_classes)]
    df1 = pd.DataFrame(columns=class_labels, index=class_labels, data=cm)
    
    # Calcular la posición del subplot
    row, col = divmod(k-1, 5)
    ax = axs[row, col]

    sns.heatmap(df1, annot=True, cmap="Greens", fmt='.0f',
                ax=ax, linewidths=3, cbar=False, annot_kws={"size": 10})
    ax.set_xlabel("Predicted Label")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("True Label")
    ax.set_title(f"K value {k}", size=10)

# Imprimir el informe de clasificación
    print(report)
    accuracy = accuracy_score(y_true_test, y_pred_test)
    print("Precisión: ", accuracy, " con valor K de: ", k)
   
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

plt.tight_layout()
plt.show()

# Crear un gráfico para comparar las precisiones en los tres conjuntos
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), precisions_train, marker='o', label='Entrenamiento')
plt.plot(range(1, 11), precisions_test, marker='o', label='Prueba')
plt.plot(range(1, 11), precisions_validation, marker='o', label='Validación')

plt.xlabel('Valor de K')
plt.ylabel('Precisión')
plt.title('Comparación de Precisión en los Conjuntos de Datos')
plt.xticks(range(1, 11))
plt.legend()
plt.grid(True)

plt.show()

# Crear gráficos para sesgo, varianza y nivel de ajuste
plt.figure(figsize=(12, 6))

# Gráfico de sesgo
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), bias_scores, marker='o', color='red')
plt.xlabel('Valor de K')
plt.ylabel('Sesgo')
plt.title('Sesgo vs. Valor de K')

# Gráfico de varianza
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), variance_scores, marker='o', color='blue')
plt.xlabel('Valor de K')
plt.ylabel('Varianza')
plt.title('Varianza vs. Valor de K')

# Crear un gráfico de nivel de ajuste
fit_levels = ["Adecuado" if abs(b) <= umbral_diferencia and abs(v) <= umbral_diferencia
              else "Underfitting" if b < 0 and v >= 0
              else "Overfitting" for b, v in zip(bias_scores, variance_scores)]

# Gráfico de nivel de ajuste
plt.figure(figsize=(8, 6))
colors = {'Adecuado': 'green', 'Underfitting': 'red', 'Overfitting': 'red'}
fit_colors = [colors[fit] for fit in fit_levels]

plt.bar(range(1, 11), bias_scores, color=fit_colors)
plt.xlabel('Valor de K')
plt.ylabel('Sesgo')
plt.title('Nivel de Ajuste vs. Valor de K. Verde = Adecuado. Rojo = Over/Underfitting')
plt.xticks(range(1, 11))
plt.grid(True)

plt.show()

print("\nEL MEJOR VALOR DE K FUE: ", best_k, "\n")
print("CON LA PRECISIÓN", best_accuracy, "\n")

# Entrenar el modelo final con el mejor valor de k encontrado
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred_final = knn_final.predict(X_test)

# Calcular la precisión en cada conjunto
accuracy_train = knn.score(X_train, y_train)
accuracy_test = accuracy_score(y_test, y_pred_final)
accuracy_validation = accuracy_score(y_val, y_pred_val)

confusion_matrix_final = confusion_matrix(y_test, y_pred_final)
df2 = pd.DataFrame(columns=class_labels, index=class_labels, data=confusion_matrix_final)
plt.figure(figsize=(8, 6))

# Crea el heatmap de la matriz de confusión del mejor valor de K.
sns.heatmap(confusion_matrix_final, annot=True, cmap="Blues", fmt="d")

# Configura los títulos y etiquetas de los ejes
plt.title(f"Confusion Matrix with best K value: {best_k}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# Valores de precisión
precisions = [accuracy_train, accuracy_test, accuracy_validation]

# Etiquetas para las barras
labels = ['Train', 'Test', 'Validation']

# Crear un gráfico de barras
plt.bar(labels, precisions, color=['blue', 'green', 'purple'])

# Agregar títulos y etiquetas de ejes
plt.title('Dataset Accuracy')
plt.xlabel('Value')
plt.ylabel('Accuracy')
plt.show()

print ("Accuracy_train: ", accuracy_train, "Accuracy_test: ", accuracy_test, "Accuracy_validation: ", accuracy_validation, "\n")

# # Umbral para considerar una diferencia significativa:
# umbral_diferencia = 0.05

# # Sesgo y Varianza: comprobar si la diferencia entre Accuracy_train y Accuracy_test es significativa
# diferencia_train_test = accuracy_train - accuracy_test
# if abs(diferencia_train_test) <= umbral_diferencia:
#     print("La diferencia entre precisión en entrenamiento y prueba es baja o moderada.\n")
# else:
#     print("La diferencia entre precisión en entrenamiento y prueba es alta (varianza).\n")

# # Comprobar si la diferencia entre Accuracy_train y Accuracy_validation es significativa
# diferencia_train_validation = accuracy_train - accuracy_validation
# if abs(diferencia_train_validation) <= umbral_diferencia:
#     print("La diferencia entre precisión en entrenamiento y validación es baja o moderada.\n")
# else:
#     print("La diferencia entre precisión en entrenamiento y validación es alta (varianza).\n")


