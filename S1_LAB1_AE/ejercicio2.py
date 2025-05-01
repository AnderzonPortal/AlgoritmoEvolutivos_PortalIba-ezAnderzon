import pandas as pd  # Importamos la biblioteca Pandas para manejo de datos estructurados

# Paso 1: Crear un diccionario con los datos de los estudiantes y sus horas usadas
datos = {
    'Estudiante': ['Ana', 'Luis', 'María', 'Juan', 'Carla'],
    'Horas_usadas': [3, 5, 2, 4, 1]
}

# Convertimos el diccionario en un DataFrame
df = pd.DataFrame(datos)

# Paso 2: Agregar una nueva columna "Costo_total", que es Horas_usadas * S/2.00
df['Costo_total'] = df['Horas_usadas'] * 2.0

# Paso 3: Mostrar el DataFrame completo con los datos y los costos
print("=========================================================")
print("Registro completo de uso del laboratorio:")
print("=========================================================")
print(df.head())  # Muestra las primeras filas del DataFrame (en este caso, todas)

# Paso 4: Calcular estadísticas descriptivas de los costos totales
estadisticas = df['Costo_total'].describe()

# Paso 5: Filtrar a los estudiantes que gastaron más de S/6.00
gasto_mayor_6 = df[df['Costo_total'] > 6.00]

# Paso 6: Imprimir el resumen final
print("=========================================================")
print(f"El gasto promedio fue de S/ {estadisticas['mean']:.2f}")
print("=========================================================")
print("Los estudiantes que gastaron más de S/6.00 son:")
print("=========================================================")
print(gasto_mayor_6['Estudiante'].tolist())
