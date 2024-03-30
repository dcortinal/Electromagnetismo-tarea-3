# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
N = 150
M = 150
dx = 0.1
dy = dx
a = int(N*dx)
b = int(M*dy)
x = np.arange(0, a, dx)
y = np.arange(0, b, dy)
X, Y = np.meshgrid(x, y)

# =============================================================================
# Método de relajación
# =============================================================================
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
    if type(x) == int:
        if x == 0 or x == a:
            return 0
    if type(y) == int:
        if y == 0:
            return -v0
        elif y == b:
            return v0

# Resuelve la ecuación de Laplace por método de relajación
V_relaxation = solve_laplace_equation(V0, N, M, dx, dy)


# =============================================================================
# Método analítico
# =============================================================================
def fourier_coeff(n):
    """
    Calcula el coeficiente de Fourier correspondiente al modo n en la expansión
    de la solución analítica de la ecuación de Laplace en un dominio rectangular
    para la función ax - x^2.
    
    Parámetros:
    n : int
        Modo de Fourier.

    Retorna:
    float
        Coeficiente de Fourier correspondiente al modo n.
    """
    return 4 * a**2 * (1 - (-1)**n)/(np.pi**3*n**3)

V_analytical = np.zeros_like(X, dtype=np.float64)  # Crea un meshgrid para los valores analíticos

for n in range(1, 21):
    V_analytical += fourier_coeff(n)/np.sinh(n*np.pi*b/a) * np.sin(n*np.pi*X/a) * np.sinh(n*np.pi*Y/a)
    V_analytical += -fourier_coeff(n)/np.sinh(n*np.pi*a/b) * np.sin(n*np.pi*X/a) * np.sinh(n*np.pi*(b-Y)/a)

# Calcula el error entre ambos métodos
error = V_analytical - V_relaxation


# =============================================================================
# Gráficas
# =============================================================================
# Figura 1: Ambas soluciones en 2D
fig1, axes1 = plt.subplots(1, 2, figsize=(22, 10))
im1 = axes1[0].imshow(V_relaxation, origin='lower', cmap='magma')
axes1[0].set_title('Método de Relajación (2D)', fontsize=30)
axes1[0].set_xlabel('X', fontsize=22)
axes1[0].set_ylabel('Y', fontsize=22)

im2 = axes1[1].imshow(V_analytical, origin='lower', cmap='magma')
axes1[1].set_title('Método Analítico (2D)', fontsize=30)
axes1[1].set_xlabel('X', fontsize=22)
axes1[1].set_ylabel('Y', fontsize=22)
fig1.colorbar(im2, ax=axes1.ravel().tolist(), shrink=0.9)

# Figura 2: Ambas soluciones en 3D
fig2 = plt.figure(figsize=(22, 10))
ax2_1 = fig2.add_subplot(1, 2, 1, projection='3d')
surf1 = ax2_1.plot_surface(X, Y, V_relaxation, cmap='magma')
ax2_1.set_title('Método de Relajación (3D)', fontsize=30)
ax2_1.set_xlabel('X', fontsize=22)
ax2_1.set_ylabel('Y', fontsize=22)
ax2_1.set_zlabel('Z', fontsize=22)

ax2_2 = fig2.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2_2.plot_surface(X, Y, V_analytical, cmap='magma')
ax2_2.set_title('Método Analítico (3D)', fontsize=30)
ax2_2.set_xlabel('X', fontsize=22)
ax2_2.set_ylabel('Y', fontsize=22)
ax2_2.set_zlabel('Z', fontsize=22)
fig2.colorbar(surf2, ax=[ax2_1, ax2_2], shrink=0.9)

# Figura 3 y 4: Error en 2D y 3D
fig3 = plt.figure(figsize=(10, 10))
im3 = plt.imshow(error, origin='lower', cmap='magma')
plt.title('Error entre Métodos (2D)', fontsize=30)
plt.xlabel('X', fontsize=22)
plt.ylabel('Y', fontsize=22)
fig3.colorbar(im3)

fig4 = plt.figure(figsize=(10, 10))
ax4 = fig4.add_subplot(111, projection='3d')
surf4 = ax4.plot_surface(X, Y, error, cmap='magma')
ax4.set_title('Error entre Métodos (3D)', fontsize=30)
ax4.set_xlabel('X', fontsize=22)
ax4.set_ylabel('Y', fontsize=22)
ax4.set_zlabel('Z', fontsize=22)
fig4.colorbar(surf4) 