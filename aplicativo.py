import numpy as np

# ==============================================
# PROBLEMA  SUMA DIRECTA EN R^4
# ==============================================

print("=" * 60)
print("PROBLEMA SUMA DIRECTA U ⊕ V ⊕ W EN R^4")
print("=" * 60)

# ==============================================
# 1. DEFINICIÓN DE LOS SUBESPACIOS Y SUS BASES
# ==============================================

print("\n1. DEFINICIÓN DE LOS SUBESPACIOS:")

# Base de U
print("\nU = {(x1,x2,x3,x4) | x1 = x2 = x4, x3 = 0}")
u_base = np.array([1, 1, 0, 1])
print(f"Base de U: B_U = {u_base}")

# Base de V
print("\nV = {(x1,x2,x3,x4) | x1 = x2 = x3, x4 = 0}")
v_base = np.array([1, 1, 1, 0])
print(f"Base de V: B_V = {v_base}")

# Base de W
print("\nW = {(x1,x2,x3,x4) | x2 = x1 + x3, 3x1 = x2 + x4}")
w_base1 = np.array([1, 0, 1, 3])
w_base2 = np.array([0, 1, -1, -1])
print(f"Base de W: B_W = {w_base1}, {w_base2}")

# ==============================================
# 2. VERIFICACIÓN DE QUE B_U ∪ B_V ∪ B_W ES BASE DE R^4
# ==============================================

print("\n" + "=" * 60)
print("2. VERIFICACIÓN DE QUE B_U ∪ B_V ∪ B_W ES BASE DE R^4")
print("=" * 60)

# Matriz con los vectores como filas
matriz_bases = np.array([
    [1, 1, 0, 1],    # u_base
    [1, 1, 1, 0],    # v_base
    [1, 0, 1, 3],    # w_base1
    [0, 1, -1, -1]   # w_base2
])

print(f"\nMatriz de los vectores base (como filas):")
print(matriz_bases)

# Verificar independencia lineal (rango = 4)
rango = np.linalg.matrix_rank(matriz_bases)
print(f"\nRango de la matriz: {rango}")
print(f"Dimensión de R^4: 4")

if rango == 4:
    print("✓ Los vectores son linealmente independientes")
    print("✓ Forman una base de R^4")
else:
    print("✗ Los vectores NO son linealmente independientes")

# Verificar que el determinante es no nulo (matriz cuadrada)
# Tomamos la matriz con vectores como columnas
matriz_columnas = matriz_bases.T
det = np.linalg.det(matriz_columnas)
print(f"\nDeterminante de la matriz (vectores como columnas): {det:.4f}")

if abs(det) > 1e-10:
    print("✓ Determinante ≠ 0 → los vectores son LI")
else:
    print("✗ Determinante = 0 → los vectores son LD")

# ==============================================
# 3. DEMOSTRACIÓN DE SUMA DIRECTA
# ==============================================

print("\n" + "=" * 60)
print("3. DEMOSTRACIÓN DE SUMA DIRECTA U ⊕ V ⊕ W")
print("=" * 60)

print("\nCondiciones para suma directa:")
print("1. R^4 = U + V + W (ya probado con la base)")
print("2. U ∩ (V + W) = {0}")
print("3. V ∩ (U + W) = {0}")
print("4. W ∩ (U + V) = {0}")

print("\nComo B_U ∪ B_V ∪ B_W es base de R^4 y tiene 4 vectores,")
print("se cumple automáticamente que R^4 = U ⊕ V ⊕ W")
print("(dimensión(U+V+W) = dim(U) + dim(V) + dim(W) = 1 + 1 + 2 = 4)")

# Verificación dimensional
dim_U = 1
dim_V = 1
dim_W = 2
dim_R4 = 4

print(f"\nDimensiones:")
print(f"dim(U) = {dim_U}")
print(f"dim(V) = {dim_V}")
print(f"dim(W) = {dim_W}")
print(f"dim(U) + dim(V) + dim(W) = {dim_U + dim_V + dim_W} = dim(R^4)")

# ==============================================
# 4. DESCOMPOSICIÓN DEL VECTOR (1,2,3,4)
# ==============================================

print("\n" + "=" * 60)
print("4. DESCOMPOSICIÓN DEL VECTOR (1, 2, 3, 4)")
print("=" * 60)

vector = np.array([1, 2, 3, 4])
print(f"\nVector a descomponer: v = {vector}")

# Resolver el sistema: M * x = vector
# Donde M es la matriz con los vectores base como columnas
M = matriz_columnas  # Ya tenemos la matriz transpuesta

# Resolver Mx = vector
coeficientes = np.linalg.solve(M, vector)
print(f"\nSistema: M * [β1, β2, β3, β4]^T = {vector}")

β1, β2, β3, β4 = coeficientes
print(f"\nCoeficientes solución:")
print(f"β1 = {β1:.4f} (coeficiente para base de U)")
print(f"β2 = {β2:.4f} (coeficiente para base de V)")
print(f"β3 = {β3:.4f} (coeficiente para primera base de W)")
print(f"β4 = {β4:.4f} (coeficiente para segunda base de W)")

# Verificación
print(f"\nVerificación:")
print(f"β1 * u_base + β2 * v_base + β3 * w_base1 + β4 * w_base2 =")
resultado = β1 * u_base + β2 * v_base + β3 * w_base1 + β4 * w_base2
print(f"{resultado} ≈ {vector} (con error: {np.linalg.norm(resultado - vector):.2e})")

# ==============================================
# 5. COMPONENTES EN CADA SUBESPACIO
# ==============================================

print("\n" + "=" * 60)
print("5. COMPONENTES EN CADA SUBESPACIO")
print("=" * 60)

# Componente en U
u_component = β1 * u_base
print(f"\nComponente en U:")
print(f"u = β1 * (1, 1, 0, 1) = {β1:.4f} * {u_base} = {u_component}")

# Componente en V
v_component = β2 * v_base
print(f"\nComponente en V:")
print(f"v = β2 * (1, 1, 1, 0) = {β2:.4f} * {v_base} = {v_component}")

# Componente en W
w_component = β3 * w_base1 + β4 * w_base2
print(f"\nComponente en W:")
print(f"w = β3 * (1, 0, 1, 3) + β4 * (0, 1, -1, -1)")
print(f"  = {β3:.4f} * {w_base1} + {β4:.4f} * {w_base2}")
print(f"  = {w_component}")

# Verificación final
print(f"\n" + "=" * 60)
print("VERIFICACIÓN FINAL: u + v + w = v_original")
print("=" * 60)
suma = u_component + v_component + w_component
print(f"\nu + v + w = {u_component} + {v_component} + {w_component}")
print(f"         = {suma}")
print(f"v_original = {vector}")
print(f"¿Coinciden? {np.allclose(suma, vector)}")

# ==============================================
# 6. PROPIEDADES UTILIZADAS
# ==============================================

print("\n" + "=" * 60)
print("PROPIEDADES Y CONCEPTOS UTILIZADOS")
print("=" * 60)

propiedades = [
    "1. Definición de subespacio por ecuaciones paramétricas/implícitas",
    "2. Obtención de bases a partir de ecuaciones",
    "3. Dimensión de subespacios",
    "4. Independencia lineal (rango de matriz)",
    "5. Suma directa de subespacios: U ⊕ V ⊕ W",
    "6. Dimensión de suma directa: dim(U⊕V⊕W) = dim(U)+dim(V)+dim(W)",
    "7. Descomposición única de vectores en suma directa",
    "8. Resolución de sistemas lineales (numpy.linalg.solve)",
    "9. Verificación de base mediante determinante/rango",
    "10. Propiedad: Si B es base de R^n, cualquier vector tiene coordenadas únicas"
]

for prop in propiedades:
    print(f"• {prop}")