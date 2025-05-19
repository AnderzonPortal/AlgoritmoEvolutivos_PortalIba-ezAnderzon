import pandas as pd
import numpy as np
import random

# Leer la matriz desde un archivo CSV
df = pd.read_csv('F:\\CICLO IX\\ALGORITMOS EVOLUTIVOS\\S5_LAB5_AE\\bd\\distancias.csv',  sep=';', index_col=0)
dist = df.values

# Calcular distancia total de una ruta
def calcular_distancia(ruta):
    return sum(dist[ruta[i], ruta[i+1]] for i in range(len(ruta) - 1))

# Generar un vecino
def generar_vecino(ruta):
    a, b = random.sample(range(len(ruta)), 2)
    nueva_ruta = ruta.copy()
    nueva_ruta[a], nueva_ruta[b] = nueva_ruta[b], nueva_ruta[a]
    return nueva_ruta

# Hill Climbing
def hill_climbing(iteraciones=1000):
    actual = list(range(len(dist)))
    random.shuffle(actual)
    mejor = actual
    mejor_dist = calcular_distancia(mejor)

    for _ in range(iteraciones):
        vecino = generar_vecino(actual)
        dist_vecino = calcular_distancia(vecino)

        if dist_vecino < mejor_dist:
            mejor = vecino
            mejor_dist = dist_vecino
            actual = vecino

    return mejor, mejor_dist

# Ejecutar
mejor_ruta, distancia_total = hill_climbing()
labs = list(df.index)
ruta_nombres = [labs[i] for i in mejor_ruta]

print('=============================================================================')
print("Orden óptimo de visita:")
print('-----------------------------------------------------------------------------')
print(" → ".join(ruta_nombres))
print('=============================================================================')
print(f"Distancia total: {distancia_total:.2f} metros")
print('=============================================================================')

