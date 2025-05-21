import pandas as pd
import random
import copy

# Leer el archivo de disponibilidad
df = pd.read_csv('F:\\CICLO IX\\ALGORITMOS EVOLUTIVOS\\S5_AVANCE_LAB5_AE\\bd\\disponibilidad.csv', sep=';')

mentores = df['MentorID'].tolist()
slots = df.columns[1:]

# Generar posibles bloques de 2 horas continuas por mentor
def bloques_disponibles(row):
    bloques = []
    for i in range(len(slots) - 1):
        if row[slots[i]] == 1 and row[slots[i+1]] == 1:
            bloques.append((i, i+1))  # índice de los slots
    return bloques

bloques_por_mentor = {}
for idx, row in df.iterrows():
    bloques_por_mentor[row['MentorID']] = bloques_disponibles(row)

# Generar una solución inicial aleatoria (1 bloque por mentor)
def generar_solucion(bloques_por_mentor):
    solucion = {}
    for mentor, bloques in bloques_por_mentor.items():
        if bloques:
            solucion[mentor] = random.choice(bloques)
    return solucion

# Calcular choques: cuántos mentores comparten el mismo bloque
def calcular_choques(solucion):
    contador = {}
    for bloque in solucion.values():
        contador[bloque] = contador.get(bloque, 0) + 1
    choques = sum(count - 1 for count in contador.values() if count > 1)
    return choques

# Búsqueda local para minimizar choques
def buscar_mejor(solucion, bloques_por_mentor):
    mejor = copy.deepcopy(solucion)
    mejor_choques = calcular_choques(mejor)

    for mentor in solucion:
        for nuevo_bloque in bloques_por_mentor[mentor]:
            if nuevo_bloque == solucion[mentor]:
                continue
            nueva_sol = copy.deepcopy(solucion)
            nueva_sol[mentor] = nuevo_bloque
            nuevos_choques = calcular_choques(nueva_sol)
            if nuevos_choques < mejor_choques:
                mejor = nueva_sol
                mejor_choques = nuevos_choques
    return mejor, mejor_choques

# Ejecutar optimización
sol = generar_solucion(bloques_por_mentor)
choques = calcular_choques(sol)

iteracion = 0
while choques > 0 and iteracion < 1000:
    nueva_sol, nuevos_choques = buscar_mejor(sol, bloques_por_mentor)
    if nuevos_choques < choques:
        sol = nueva_sol
        choques = nuevos_choques
    else:
        break
    iteracion += 1

# Mostrar resultado final
print('=============================================================================')
print("Asignación final de bloques:")
print('-----------------------------------------------------------------------------')
for mentor in sorted(sol):
    print(f"{mentor}: Slot{sol[mentor][0]+1} y Slot{sol[mentor][1]+1}")
    
print('=============================================================================')
print("Choques:", choques)
print('=============================================================================')