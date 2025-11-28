import numpy as np
from scipy.linalg import null_space, orth
# Definir un conjunto de vectores en R^3
vectores = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print("Vectores originales:")
print(vectores)

# Calcular el rango (dimension del espacio columna)
rango = np.linalg.matrix_rank(vectores)
print(f"\nRango de la matriz (dimension del espacio columna): {rango}")

# Encontrar una base para el espacio columna
base_espacio_columna = orth(vectores)
print(f"\nBase del espacio columna (ortonormal):")
print(base_espacio_columna)

# Encontrar una base para el espacio nulo
base_espacio_nulo = null_space(vectores)
print(f"\nBase del espacio nulo:")
print(base_espacio_nulo)

# Verificar si los vectores son linealmente independientes
determinante = np.linalg.det(vectores[:rango, :rango])
print(f"\nDeterminante del subconjunto principal: {determinante}")

if abs(determinante) > 1e-10:
    print("Los vectores son linealmente independientes en el subespacio")
else:
    print("Los vectores son linealmente dependientes")

# Proyeccion ortogonal sobre el subespacio generado
def proyeccion_ortogonal(vector, base):
#Calcula la proyeccion ortogonal de un vector sobre un subespacio
    proyeccion = np.zeros_like(vector, dtype=float)
    for vector_base in base.T:
        proyeccion += np.dot(vector, vector_base) * vector_base
    return proyeccion

vector_test = np.array([1, 0, 0])
proyeccion = proyeccion_ortogonal(vector_test, base_espacio_columna)
print(f"\nProyeccion del vector [1, 0, 0] sobre el subespacio: {proyeccion}")