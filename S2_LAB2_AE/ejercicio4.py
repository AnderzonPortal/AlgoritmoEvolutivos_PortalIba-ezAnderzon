import pandas as pd

gastos = [4.0, 3.5, 5.0, 4.2, 3.8]
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
df_gastos = pd.DataFrame({'Día': dias, 'Gasto': gastos})

total = df_gastos['Gasto'].sum()
promedio = df_gastos['Gasto'].mean()
mayores_al_promedio = df_gastos[df_gastos['Gasto'] > promedio]
print("=============================================")
print("Gasto total:", total)
print("Gasto promedio:", promedio)
print("=============================================")
print("Días con gasto mayor al promedio:")
print("---------------------------------")
print(mayores_al_promedio)
