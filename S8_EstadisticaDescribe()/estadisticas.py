import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

file_path = 'S7_notas_1u.csv'
data = pd.read_csv(file_path)

#Tabla del describe de estudiantes
desc_stats = data.groupby('Tipo_Examen')['Nota'].describe()
print("Estadísticas descriptivas por tipo de examen:")
print(tabulate(desc_stats, headers='keys', tablefmt='fancy_grid', showindex=True))

colors = ['#9C5E45', "#359AD4", "#2ACE38"]  # Tonos cálidos y naturales

# Gráficos de Caja
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Tipo_Examen', y='Nota', palette=colors) #Comando Seaborn
plt.title('Distribución de las notas por tipo de examen', fontsize=14)
plt.xlabel('Tipo de Examen', fontsize=12)
plt.ylabel('Nota', fontsize=12)
plt.xticks(rotation=0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Gráfico de histograma
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Nota', hue='Tipo_Examen', kde=True, multiple='stack', palette=colors)
plt.title('Distribución de las notas por tipo de examen (Histograma)', fontsize=14)
plt.xlabel('Nota', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xticks(range(8, int(data['Nota'].max()) + 1, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
