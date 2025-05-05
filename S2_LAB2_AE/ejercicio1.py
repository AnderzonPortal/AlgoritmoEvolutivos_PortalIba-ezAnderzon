import numpy as np

presupuesto = 8
precios = np.array([0.10, 0.12, 0.08])
paginas = np.floor(presupuesto / precios)
mejor_opcion = np.argmax(paginas)
print("=============================================")
print("Páginas por copistería:", paginas)
#Se suma 1 porque el indice empieza en 0*//
print("Mejor opción (más páginas): Copistería", mejor_opcion + 1) 
print("=============================================")
