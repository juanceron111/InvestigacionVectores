import numpy as np

def producto_interior_personalizado(u, v, matriz_A):
    """
    Define un producto interior personalizado: <u, v> = u^T A v
    donde A es una matriz simétrica y definida positiva
    """
    return np.dot(u.T, np.dot(matriz_A, v))


def gram_schmidt(vectores, producto_interior):
    """
    Implementa el proceso de ortogonalización de Gram-Schmidt
    con un producto interior personalizado
    """
    n = len(vectores)
    base_ortogonal = []

    for i in range(n):
        v_actual = vectores[i].copy().astype(float)

        # Restar las proyecciones sobre los vectores anteriores
        for j in range(i):
            numerador = producto_interior(v_actual, base_ortogonal[j])
            denominador = producto_interior(base_ortogonal[j], base_ortogonal[j])
            proyeccion = (numerador / denominador) * base_ortogonal[j]
            v_actual -= proyeccion

        # Normalizar el vector
        norma = np.sqrt(producto_interior(v_actual, v_actual))
        if norma > 1e-10:
            v_actual /= norma

        base_ortogonal.append(v_actual)

    return base_ortogonal


# Ejemplo de uso
# Definir una matriz simétrica y definida positiva para el producto interior
A = np.array([
    [2, 1, 0],
    [1, 2, 1],
    [0, 1, 2]
])

# Verificar que A es definida positiva
autovalores = np.linalg.eigvals(A)
print(f"Autovalores de A: {autovalores}")
print("A es definida positiva:", np.all(autovalores > 0))

# Definir vectores linealmente independientes
vectores_originales = [
    np.array([1, 0, 0]),
    np.array([1, 1, 0]),
    np.array([1, 1, 1])
]

# Crear función de producto interior con matriz A
def producto_inner(u, v):
    return producto_interior_personalizado(u, v, A)

# Aplicar Gram-Schmidt
base_ortonormal = gram_schmidt(vectores_originales, producto_inner)

print("\nBase ortonormal obtenida:")
for i, vector in enumerate(base_ortonormal):
    print(f"v{i+1}: {vector}")

# Verificación de ortogonalidad
print("\nVerificación de ortogonalidad:")
for i in range(len(base_ortonormal)):
    for j in range(i + 1, len(base_ortonormal)):
        producto = producto_inner(base_ortonormal[i], base_ortonormal[j])
        print(f"<v{i+1}, v{j+1}> = {producto:.6f}")

# Verificación de normas
print("\nVerificación de normas:")
for i, vector in enumerate(base_ortonormal):
    norma = np.sqrt(producto_inner(vector, vector))
    print(f"||v{i+1}|| = {norma:.6f}")
