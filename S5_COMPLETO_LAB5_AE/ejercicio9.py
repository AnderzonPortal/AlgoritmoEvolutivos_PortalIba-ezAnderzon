import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

from deap import base, creator, tools

# 1. Cargar datos
df = pd.read_csv('E:\\CICLO IX\\ALGORITMOS EVOLUTIVOS\\S5_AVANCE_LAB5_AE\\bd\\emails.csv', sep=';')

X = df.iloc[:, :5].values
y = df['Spam'].values

# Dividir datos en train y validación (evolución solo evalúa en validación)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Configuración DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximizar F1
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Genotipo: 5 pesos y 1 umbral
# Pesos entre [0, 1], umbral entre [0, sum(weights)]
def init_individual():
    weights = [random.uniform(0, 1) for _ in range(5)]
    threshold = random.uniform(0, 5)  # un rango suficiente para el umbral
    return creator.Individual(weights + [threshold])

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3. Función para predecir usando pesos y umbral
def predict(individual, X):
    weights = np.array(individual[:5])
    threshold = individual[5]
    scores = np.dot(X, weights)
    return (scores >= threshold).astype(int)

# 4. Evaluación = F1 sobre validación
def eval_individual(individual):
    y_pred = predict(individual, X_val)
    score = f1_score(y_val, y_pred)
    return (score,)

# 5. Mutación: añadir ruido gaussiano a cada gen (peso o umbral)
def mutacion(individual, mu=0, sigma=0.1, indpb=0.3):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
            # limitar pesos a [0,1]
            if i < 5:
                individual[i] = max(0, min(1, individual[i]))
            else:
                # umbral limitado en [0, 5]
                individual[i] = max(0, min(5, individual[i]))
    return (individual,)

# 6. Hill climbing local: mejora individual mutándolo localmente varias veces y tomando mejor
def hill_climbing_local(individual, iterations=10):
    best = creator.Individual(individual)
    best.fitness.values = eval_individual(best)
    for _ in range(iterations):
        neighbor = creator.Individual(best)
        mutacion(neighbor, sigma=0.05, indpb=0.5)
        neighbor.fitness.values = eval_individual(neighbor)
        if neighbor.fitness.values[0] > best.fitness.values[0]:
            best = neighbor
    return best

# 7. Función de evaluación con hill climbing local
def eval_with_hill_climbing(individual):
    # Mutar el individuo
    mutacion(individual)
    # Aplicar hill climbing local para mejorar
    improved = hill_climbing_local(individual, iterations=10)
    # Copiar mejora al original para que la población evolucione mejor
    for i in range(len(individual)):
        individual[i] = improved[i]
    return improved.fitness.values

toolbox.register("evaluate", eval_with_hill_climbing)
toolbox.register("mutate", mutacion)
toolbox.register("select", tools.selBest)

def main():
    random.seed(42)
    pop = toolbox.population(n=20)
    ngen = 30
    stats = []
    best_individuals = []

    for gen in range(ngen):
        # Evaluar todos
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Selección: escoger mejores 10
        selected = toolbox.select(pop, 10)
        
        # Generar nueva población con mutación e hill climbing local
        offspring = []
        while len(offspring) < 20:
            parent = random.choice(selected)
            child = creator.Individual(parent)
            toolbox.mutate(child)
            child.fitness.values = toolbox.evaluate(child)
            offspring.append(child)
        
        pop = offspring
        
        # Guardar mejor fitness y mejor individuo
        fits = [ind.fitness.values[0] for ind in pop]
        best_idx = np.argmax(fits)
        best_fit = fits[best_idx]
        best_ind = pop[best_idx]
        
        best_individuals.append(best_ind)
        stats.append(best_fit)
        
        print(f"Gen {gen+1} - Mejor F1: {best_fit:.4f}")
    
    # Mejor individuo final
    best_idx = np.argmax(stats)
    best = best_individuals[best_idx]
    print("\nMejor solución encontrada:")
    print("Pesos:", best[:5])
    print("Umbral:", best[5])
    print(f"F1-score: {stats[best_idx]:.4f}")
    
    # Gráfica
    plt.plot(stats)
    plt.xlabel("Generación")
    plt.ylabel("Mejor F1-score")
    plt.title("Evolución F1-score")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
