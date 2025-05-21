import pandas as pd
import numpy as np
import random

# 1. Cargar el dataset
df = pd.read_csv('F:\\CICLO IX\\ALGORITMOS EVOLUTIVOS\\S5_AVANCE_LAB5_AE\\bd\\notas.csv' , sep=';')

# 2. Función de aptitud
def fitness(offset, data):
    adjusted = data[['Parcial1', 'Parcial2', 'Parcial3']] + offset
    adjusted = adjusted.clip(0, 20)
    final_avg = adjusted.mean(axis=1)
    class_avg = final_avg.mean()
    
    approved_pct = (final_avg >= 11).sum() / len(final_avg)
    
    # Penalización si se supera el promedio permitido
    if class_avg > 14:
        return approved_pct - (class_avg - 14) * 0.5
    return approved_pct

# 3. Hill Climbing
best_offset = 0
best_score = fitness(0, df)

for _ in range(1000):  # 1000 iteraciones
    candidate = round(random.uniform(-5, 5) * 2) / 2  # valores entre -5 y 5 en pasos de 0.5
    score = fitness(candidate, df)
    if score > best_score:
        best_score = score
        best_offset = candidate

# 4. Resultado
adjusted_df = df.copy()
for col in ['Parcial1', 'Parcial2', 'Parcial3']:
    adjusted_df[col] = (adjusted_df[col] + best_offset).clip(0, 20)
adjusted_df['Promedio'] = adjusted_df[['Parcial1', 'Parcial2', 'Parcial3']].mean(axis=1)

print('=============================================================================')
print(f'Mejor offset encontrado: {best_offset}')
print('=============================================================================')
print(f'Porcentaje de aprobados: {(adjusted_df["Promedio"] >= 11).mean() * 100:.2f}%')
print('=============================================================================')
print(f'Promedio general: {adjusted_df["Promedio"].mean():.2f}')
print('=============================================================================')

