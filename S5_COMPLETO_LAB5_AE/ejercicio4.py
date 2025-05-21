import pandas as pd
import random
import numpy as np

# Leer los datos
df = pd.read_csv('F:\\CICLO IX\\ALGORITMOS EVOLUTIVOS\\S5_AVANCE_LAB5_AE\\bd\\proyectos.csv',  sep=';')

# Constantes
PRESUPUESTO = 10000
N = len(df)

# Función de aptitud
def aptitud(bitstring):
    costo_total = sum(df.loc[i, 'Cost_Soles'] for i in range(N) if bitstring[i] == 1)
    if costo_total > PRESUPUESTO:
        return -float('inf')
    return sum(df.loc[i, 'Benefit_Soles'] for i in range(N) if bitstring[i] == 1)

# Generar vecino
def vecino(bitstring):
    vecino = bitstring.copy()
    i = random.randint(0, N - 1)
    vecino[i] = 1 - vecino[i]  # Voltear bit
    return vecino

# Hill climbing
def hill_climbing(iteraciones=1000):
    actual = [random.randint(0, 1) for _ in range(N)]
    mejor = actual
    mejor_aptitud = aptitud(mejor)

    for _ in range(iteraciones):
        v = vecino(actual)
        a = aptitud(v)
        if a > mejor_aptitud:
            mejor = v
            mejor_aptitud = a
            actual = v

    return mejor, mejor_aptitud

# Ejecutar
solucion, beneficio = hill_climbing()

# Mostrar resultados
seleccionados = df[[bool(b) for b in solucion]]
print('=============================================================================')
print("Proyectos seleccionados:")
print('=============================================================================')
print(seleccionados[['ProjectID', 'Cost_Soles', 'Benefit_Soles']])
print('----------------------------------------------------------')
print(f"\nBeneficio total: S/ {beneficio}")
print('=============================================================================')
print(f"Costo total: S/ {seleccionados['Cost_Soles'].sum()}")
print('=============================================================================')
