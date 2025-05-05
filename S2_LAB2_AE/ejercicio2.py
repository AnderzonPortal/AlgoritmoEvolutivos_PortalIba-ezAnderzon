import numpy as np

presupuesto = 15
precios = np.array([2.50, 3.00, 1.80])


viajes = np.floor(presupuesto / precios)

max_viajes = viajes.max()
mejor_opcion = viajes.argmax()

print("=============================================")
print("Viajes posibles:", viajes)
print("Máximo de viajes que puede hacer:", max_viajes)
print("Medio más conveniente:", mejor_opcion +1)
print("=============================================")
