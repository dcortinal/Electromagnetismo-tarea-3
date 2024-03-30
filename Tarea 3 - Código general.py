import numpy as np
import matplotlib.pyplot as plt

# Parámetros
N = int(input("Por favor ingrese el número de grillas horizontales (N): "))
M = int(input("Por favor ingrese el número de grillas verticales (M): "))
dx = float(input("Por favor ingresa el valor del paso (dx = dy): "))
dy = dx
a = int(N*dx)
b = int(M*dy)

def solve_laplace_equation(V0, N, M, dx, dy, tolerance=1e-6):
    """
    Se resuelve la ecuación de Laplace en un dominio rectangular.

    Parámetros:
    V0 : función
        Función de las condiciones de frontera.
    N : int
        Número de pasos en el eje x.
    M : int
        Número de pasos en el eje y.
    dx : float
        Tamaño del paso en el eje x.
    dy : float
        Tamaño del paso en el eje y.
    tolerance : float, opcional
        Tolerancia para la convergencia del método iterativo.

    Retorna:
    numpy.ndarray
        La matriz de potenciales después de la convergencia.
    """

    # Inicializa la matriz de potenciales con ceros
    V = np.zeros((M, N))

    # Aplica las condiciones de frontera
    V[:, 0] = V0(0, np.linspace(0, b, M))
    V[:, -1] = V0(a, np.linspace(0, b, M))
    V[0, :] = V0(np.linspace(0, a, N), 0)
    V[-1, :] = V0(np.linspace(0, a, N), b)

    max_diff = 1  # Diferencia inicial

    # Itera hasta que la diferencia entre dos iteraciones consecutivas sea menor que la tolerancia
    while max_diff > tolerance:
        V_old = V.copy()
        for i in range(1, N - 1):
            for j in range(1, M - 1):
                V[i, j] = (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1]) / 4
        max_diff = np.max(np.abs(V - V_old))

    return V

def V0(x, y):
    """
    Define la condición de frontera para la ecuación de Laplace.

    Parámetros:
    x : int o array-like
        Coordenada x.
    y : int o array-like
        Coordenada y.

    Retorna:
    float o array-like
        Valor del potencial en el punto dado por las coordenadas (x, y).
    """
    v0 = a*x - x**2
    return v0


# Resuelve la ecuación de Laplace
V = solve_laplace_equation(V0, N, M, dx, dy)

# Genera la representación gráfica de la solución
fig = plt.figure(figsize=(10, 10))
plt.title("Solución general", fontsize=28)
plt.xlabel("X", fontsize=22)
plt.ylabel("Y", fontsize=22)
plt.imshow(V, cmap="magma")
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()