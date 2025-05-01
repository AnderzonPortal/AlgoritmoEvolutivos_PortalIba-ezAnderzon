import numpy as np  # Importamos la biblioteca NumPy para cálculos numéricos

# Paso 1: Crear un array con los precios del café en las cuatro cafeterías
precios = np.array([2.50, 3.00, 1.75, 2.20])  # A, B, C, D respectivamente

# Paso 2: Calcular cuántos cafés puede comprar Jorge en cada cafetería con S/10
max_cafes = np.floor(10 / precios)  # División vectorizada + redondeo hacia abajo

# Paso 3: Encontrar la mayor cantidad de cafés posibles y el índice correspondiente
cantidad_maxima = int(max_cafes.max())         # Convertimos a entero por claridad
indice_maximo = max_cafes.argmax()             # Índice donde ocurre esa cantidad

# Paso 4: Obtener el precio mínimo y su índice
precio_minimo = precios.min()
indice_precio_minimo = precios.argmin()

# Nombres de las cafeterías para referencia humana
cafeterias = ['A', 'B', 'C', 'D']

# Paso 5: Mostrar los resultados de manera clara
print("Con S/10 puedo comprar:")
for i in range(len(precios)):
    print(f"- {int(max_cafes[i])} cafés en la cafetería {cafeterias[i]} (precio S/{precios[i]:.2f})")

print(f"\nLa mayor cantidad de cafés que puedo comprar es {cantidad_maxima} en la cafetería {cafeterias[indice_maximo]}.")
print(f"El precio mínimo es S/{precio_minimo:.2f}, en la cafetería {cafeterias[indice_precio_minimo]}.")
