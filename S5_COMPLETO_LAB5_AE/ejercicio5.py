import pandas as pd
import random
import numpy as np
from collections import defaultdict

# Cargar datos
df = pd.read_csv('F:\\CICLO IX\\ALGORITMOS EVOLUTIVOS\\S5_AVANCE_LAB5_AE\\bd\\tesistas.csv',  sep=';')
franjas = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
n_salas = 6
n_franjas = len(franjas)
n_tesistas = len(df)

# Crear estructura: disponibilidad
disponibilidad = {
    row['TesistaID']: [i for i, f in enumerate(franjas) if row[f] == 1]
    for _, row in df.iterrows()
}

# Asignación inicial heurística (secuencial)
def asignacion_inicial():
    asignacion = {}
    sala_fr = defaultdict(set)  # salas ocupadas en cada franja
    for tesista in df['TesistaID']:
        for f in disponibilidad[tesista]:
            for sala in range(n_salas):
                if (sala, f) not in sala_fr:
                    asignacion[tesista] = (sala, f)
                    sala_fr[(sala, f)] = tesista
                    break
            else:
                continue
            break
    return asignacion

# Costo: penaliza solapamientos y franjas excesivas
def costo(asignacion):
    solapamientos = 0
    uso_salas = defaultdict(list)

    for tesista, (sala, franja) in asignacion.items():
        uso_salas[sala].append(franja)

    for sala, franjas_usadas in uso_salas.items():
        if len(franjas_usadas) != len(set(franjas_usadas)):
            solapamientos += len(franjas_usadas) - len(set(franjas_usadas))
        franjas_usadas.sort()
        bloques = np.diff(franjas_usadas)
        huecos = sum(b > 1 for b in bloques)
        exceso = max(0, len(franjas_usadas) - 4)
        solapamientos += huecos + exceso

    return solapamientos

# Generar vecino
def vecino(asignacion):
    nuevo = asignacion.copy()
    tesista = random.choice(list(nuevo.keys()))
    disponibles = disponibilidad[tesista]
    nueva_franja = random.choice(disponibles)
    nueva_sala = random.randint(0, n_salas - 1)
    nuevo[tesista] = (nueva_sala, nueva_franja)
    return nuevo

# Hill climbing
def hill_climbing(iteraciones=1000):
    actual = asignacion_inicial()
    mejor = actual
    mejor_costo = costo(mejor)

    for _ in range(iteraciones):
        vecino_asig = vecino(actual)
        c = costo(vecino_asig)
        if c < mejor_costo:
            mejor = vecino_asig
            mejor_costo = c
            actual = vecino_asig

    return mejor, mejor_costo

# Ejecutar
asignacion_final, costo_final = hill_climbing()

# Mostrar resultados
calendario = pd.DataFrame([
    {"Tesista": t, "Sala": s+1, "Franja": f+1}
    for t, (s, f) in asignacion_final.items()
])
calendario = calendario.sort_values(by=["Franja", "Sala"])

print('=============================================================================')
print("Calendario final de defensas:")
print('-----------------------------------------------------------------------------')
print(calendario.to_string(index=False))
print('=============================================================================')
print(f"\nMétrica de conflictos (solapamientos + huecos + excesos): {costo_final}")
print('=============================================================================')