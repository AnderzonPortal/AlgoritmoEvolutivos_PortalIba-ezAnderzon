import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from deap import base, creator, tools
from math import sqrt
import random

# 1. Cargar y preparar los datos
df = pd.read_csv('E:\\CICLO IX\\ALGORITMOS EVOLUTIVOS\\S5_AVANCE_LAB5_AE\\bd\\HousePricesUNS.csv', sep=';')
X = df[['Rooms', 'Area_m2']].values
y = df['Price_Soles'].values

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Crear entorno DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_alpha", random.uniform, 0.001, 1000.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_alpha, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3. Evaluación segura con alpha mínimo 0.001
def evaluate(individual):
    alpha = max(individual[0], 0.001)  # Asegura que alpha ≥ 0.001
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)
    return (rmse,)

toolbox.register("evaluate", evaluate)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10.0, indpb=1.0)
toolbox.register("select", tools.selBest, k=1)

# 4. Hill climbing
def hill_climbing(pop_size=20, generations=50):
    pop = toolbox.population(n=pop_size)
    log_rmse = []

    for gen in range(generations):
        offspring = []
        for ind in pop:
            clone = toolbox.clone(ind)
            toolbox.mutate(clone)

            # Asegura que alpha no sea negativo después de mutar
            clone[0] = max(clone[0], 0.001)

            clone.fitness.values = toolbox.evaluate(clone)

            if clone.fitness.values[0] < ind.fitness.values[0] if ind.fitness.valid else True:
                offspring.append(clone)
            else:
                offspring.append(ind)

        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        best = tools.selBest(pop, 1)[0]
        log_rmse.append(best.fitness.values[0])
        print(f"Gen {gen+1}: Best α = {best[0]:.4f} | RMSE = {best.fitness.values[0]:.4f}")

    return best, log_rmse

# 5. Ejecutar
best_individual, curva = hill_climbing()

print(f"\nα óptimo: {best_individual[0]:.4f}")
plt.plot(curva)
plt.title("Curva de Convergencia - RMSE")
plt.xlabel("Generación")
plt.ylabel("RMSE")
plt.grid(True)
plt.show()
