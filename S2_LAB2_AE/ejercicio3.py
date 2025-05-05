import pandas as pd

data = {
    'Estudiante': ['Rosa', 'David', 'Elena', 'Mario', 'Paula'],
    'Días_prestamo': [7, 10, 5, 12, 3]
}
df = pd.DataFrame(data)
print("=============================================")
print(df['Días_prestamo'].describe())
print("=============================================")
print("Días máximo de préstamo:", df['Días_prestamo'].max())
print("=============================================")
print("Estudiantes con más de 8 días:")
print("-----------------------------")
print(df[df['Días_prestamo'] > 8])
print("=============================================")