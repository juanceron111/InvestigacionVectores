import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Generar datos simulados de clientes
# Características: [ingreso_anual, frecuencia_compra]
X, y = make_blobs(
    n_samples=200,
    centers=2,
    n_features=2,
    random_state=42,
    cluster_std=2.0
)

# Escalar los datos (normalización)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar un clasificador SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_scaled, y)


# -----------------------------
#   VISUALIZACIÓN DEL MODELO
# -----------------------------
def visualizar_svm(X, y, modelo, titulo):
    plt.figure(figsize=(10, 8))

    # Crear malla
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predecir sobre la malla
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Contorno de clasificación
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

    # Datos
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=y,
        edgecolors='k', marker='o', s=50,
        cmap=plt.cm.coolwarm
    )

    # Vectores de soporte
    plt.scatter(
        modelo.support_vectors_[:, 0],
        modelo.support_vectors_[:, 1],
        s=100, facecolors='none', edgecolors='k'
    )

    # Estética
    plt.title(titulo)
    plt.xlabel("Característica 1 (Ingreso escalado)")
    plt.ylabel("Característica 2 (Frecuencia escalada)")
    plt.legend(*scatter.legend_elements(), title="Clases")
    plt.grid(True, alpha=0.3)
    plt.show()


visualizar_svm(X_scaled, y, svm, "Clasificación de Clientes usando SVM")


# -----------------------------
#   ANÁLISIS DEL MODELO
# -----------------------------
print("Información del modelo SVM:")
print(f"Número de vectores de soporte: {len(svm.support_vectors_)}")
print(f"Coeficientes del hiperplano: {svm.coef_}")
print(f"Intercepto: {svm.intercept_}")

# Calcular margen
w_norm = np.linalg.norm(svm.coef_)
margen = 2 / w_norm
print(f"Margen del clasificador: {margen:.4f}")


# -----------------------------
#   CLASIFICAR NUEVOS CLIENTES
# -----------------------------
def clasificar_cliente(ingreso, frecuencia, modelo, scaler):
    cliente = np.array([[ingreso, frecuencia]])
    cliente_escalado = scaler.transform(cliente)

    prediccion = modelo.predict(cliente_escalado)
    decision = modelo.decision_function(cliente_escalado)

    clase = "Cliente Premium" if prediccion[0] == 1 else "Cliente Estándar"

    # Convertir margen a una "probabilidad" mediante sigmoide
    confianza = 1 / (1 + np.exp(-decision[0]))

    return clase, confianza


# Ejemplo
ingreso_ejemplo = 75000
frecuencia_ejemplo = 12

clase, confianza = clasificar_cliente(
    ingreso_ejemplo, frecuencia_ejemplo, svm, scaler
)

print("\nClasificación de cliente ejemplo:")
print(f"Ingreso: ${ingreso_ejemplo:,}, Frecuencia: {frecuencia_ejemplo} compras/año")
print(f"Clasificación: {clase}")
print(f"Confianza: {confianza:.4f}")
