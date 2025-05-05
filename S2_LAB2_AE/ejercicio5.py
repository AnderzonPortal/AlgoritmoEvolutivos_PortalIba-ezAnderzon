import numpy as np

gb = np.array([1, 2, 5, 10])        
precios = np.array([5, 9, 20, 35])    

costo_por_gb = precios / gb

costo_minimo = np.min(costo_por_gb)
indice_mejor_paquete = np.argmin(costo_por_gb)

print("==================================================")
print("Costo por GB para cada paquete:", costo_por_gb)
print("Costo mínimo por GB:", costo_minimo)
print("Mejor paquete (índice):", indice_mejor_paquete)
print("==================================================")