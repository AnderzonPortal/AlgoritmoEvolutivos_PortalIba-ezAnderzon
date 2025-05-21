import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deap import base, creator, tools
import random
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Carga y preprocesamiento ---
df = pd.read_csv('E:\\CICLO IX\\ALGORITMOS EVOLUTIVOS\\S5_AVANCE_LAB5_AE\\bd\\enrollments.csv', sep=';')

# Variables
X = df[['Credits', 'Prev_GPA', 'Extracurricular_hours']].values

# Codificamos la categoría
le = LabelEncoder()
y = le.fit_transform(df['Category'])  # Baja=0, Media=1, Alta=2

# Escalamos características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividimos datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertimos a tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

input_size = X_train.shape[1]
output_size = 3  # tres clases

# --- 2. Definimos la red neuronal parametrizable ---

class Net(nn.Module):
    def __init__(self, layers, neurons):
        super(Net, self).__init__()
        # layers = número de capas ocultas
        # neurons = lista con neuronas por capa
        layer_list = []
        last_size = input_size
        for i in range(layers):
            layer_list.append(nn.Linear(last_size, neurons[i]))
            layer_list.append(nn.ReLU())
            last_size = neurons[i]
        layer_list.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layer_list)
    def forward(self, x):
        return self.net(x)

# --- 3. Fitness: entrena max 20 epochs y calcula accuracy ---

def eval_nn(individual):
    # individual = [num_layers(int), neurons1(int), neurons2(int), neurons3(int), lr(float)]
    layers = individual[0]
    neurons = individual[1:4][:layers]
    lr = individual[4]

    # Construimos la red
    net = Net(layers, neurons)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Entrenamiento corto (20 epochs)
    net.train()
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = net(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluación
    net.eval()
    with torch.no_grad():
        outputs = net(X_test_t)
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(y_test, predicted.numpy())

    # Queremos maximizar accuracy
    return (acc,)

# --- 4. Definimos el genotipo y herramientas DEAP ---

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Genotipo: 5 genes:
# 0: capas: 1-3 (int)
# 1-3: neuronas capa 1-3: 5-20 (int)
# 4: lr: 0.001-0.1 (float)

toolbox.register("num_layers", random.randint, 1, 3)
toolbox.register("neurons", random.randint, 5, 20)
toolbox.register("lr", random.uniform, 0.001, 0.1)

def create_individual():
    layers = toolbox.num_layers()
    neurons = [toolbox.neurons() for _ in range(3)]
    lr = toolbox.lr()
    return creator.Individual([layers] + neurons + [lr])

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_nn)

# Mutación: muta ligeramente neuronas o lr, o cambia capas

def mutate(individual):
    gene = random.randint(0,4)
    if gene == 0:
        # cambio en capas
        individual[0] = random.randint(1,3)
    elif gene in [1,2,3]:
        # pequeña mutación en neuronas +/-1 manteniendo rango
        old = individual[gene]
        change = random.choice([-1,1])
        new_val = min(20, max(5, old + change))
        individual[gene] = new_val
    else:
        # mutación pequeña en lr (±0.01)
        old = individual[4]
        change = random.uniform(-0.01, 0.01)
        new_val = min(0.1, max(0.001, old + change))
        individual[4] = new_val
    return individual,

toolbox.register("mutate", mutate)

# Hill climbing local (pequeños ajustes similares a mutación)

def hill_climbing(individual):
    best = individual[:]
    best_fit = eval_nn(best)[0]
    for _ in range(5):
        neighbor = best[:]
        # mutamos una vez
        neighbor, = mutate(neighbor)
        fit = eval_nn(neighbor)[0]
        if fit > best_fit:
            best = neighbor
            best_fit = fit
    return best

# Selección tipo torneo

toolbox.register("select", tools.selTournament, tournsize=3)

# --- 5. Algoritmo evolutivo simple con hill climbing local ---

def main():
    pop = toolbox.population(n=10)
    NGEN = 10

    # Evaluar población inicial
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for gen in range(NGEN):
        print('================================================')
        print(f"-- Generación {gen} --")
        print('================================================')
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Mutar y hill climbing local
        for ind in offspring:
            if random.random() < 0.7:
                toolbox.mutate(ind)
            # Mejora local con hill climbing
            improved = hill_climbing(ind)
            ind[:] = improved[:]
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Mejor accuracy en generación {gen}: {max(fits)}")

    best_ind = tools.selBest(pop, 1)[0]
    print('================================================')
    print("Mejor individuo:", best_ind)
    print('================================================')
    print("Arquitectura final:")
    print('================================================')
    print(f"Capas: {best_ind[0]}")
    print('================================================')
    print(f"Neurona(s) por capa: {best_ind[1:1+best_ind[0]]}")
    print('================================================')
    print(f"Tasa de aprendizaje: {best_ind[4]:.4f}")

    # Evaluar con mejor individuo y mostrar accuracy final
    final_acc = eval_nn(best_ind)[0]
    print(f"Accuracy final: {final_acc:.4f}")

if __name__ == "__main__":
    main()
