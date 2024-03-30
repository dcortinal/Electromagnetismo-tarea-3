# Cálculo del potencial en una grilla bidimensional por método de relajación

Este programa consiste en calcular el potencial eléctrico en una grilla bidimensional de tamaño N×M con condiciones de frontera de Dirichlet utilizando dos métodos: el método de relajación y el método analítico. El objetivo es comparar los resultados obtenidos por ambos métodos y analizar el error entre ellos.

## Contenido del Repositorio

- `Tarea 3 - Código general.py`: Script principal que implementa el método de relajación para calcular el potencial en la grilla bidimensional.
- `Tarea 3 - Código condiciones.py`: Script adicional que implementa el método analítico para calcular el potencial en la misma grilla bidimensional y compararlo con el método de relajación bajo unas condiciones de Dirichlet específicas.
- `README.md`: Este archivo que proporciona una visión general del proyecto y las instrucciones de uso.

## Requisitos

- Python 3.x
- Bibliotecas NumPy y Matplotlib

## Uso

1. Clona este repositorio localmente:

```bash
git clone https://github.com/dcortinal/Electromagnetismo-tarea-3.git
cd Electromagnetismo-tarea-3
python Tarea 3 - Código general.py
```
2. Al ejecutar `Tarea 3 - Código general.py`, el programa le pedirá el tamaño de N, M y de los pasos dx y dy con tres inputs.
   
3. Si desea cambiar las condiciones de frontera, deberá hacer los cambios pertinentes en la función V0, lo demás deberá permanecer sin alteraciones. Un ejemplo de esto se encuentra en el archivo `Tarea 3 - Código condiciones.py`.

4. Una vez finalizado el cálculo, se mostrará una representación gráfica de la solución en una ventana emergente.

5. Si desea comparar los resultados con el método analítico, puede ejecutar el script `Tarea 3 - Código condiciones.py` de manera similar.
