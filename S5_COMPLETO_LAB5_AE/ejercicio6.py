import pandas as pd
import random

# Carga el archivo CSV
df = pd.read_csv('F:\\CICLO IX\\ALGORITMOS EVOLUTIVOS\\S5_AVANCE_LAB5_AE\\bd\\preguntas.csv',  sep=';')

questions = df.to_dict('records')  # lista de dicts: [{"QuestionID":..., "Difficulty":..., "Time_min":...}, ...]

MAX_TIME = 90
MIN_DIFF = 180
MAX_DIFF = 200
TARGET_DIFF = (MIN_DIFF + MAX_DIFF) / 2

def evaluate(state):
    total_time = sum(q["Time_min"] for q,s in zip(questions,state) if s==1)
    total_diff = sum(q["Difficulty"] for q,s in zip(questions,state) if s==1)

    penalty_time = 0
    if total_time > MAX_TIME:
        penalty_time = (total_time - MAX_TIME) * 10

    penalty_diff = 0
    if total_diff < MIN_DIFF:
        penalty_diff = (MIN_DIFF - total_diff) * 10
    elif total_diff > MAX_DIFF:
        penalty_diff = (total_diff - MAX_DIFF) * 10

    cost = penalty_time + penalty_diff + abs(total_diff - TARGET_DIFF)
    return cost, total_time, total_diff

def get_neighbors(state):
    neighbors = []
    for i in range(len(state)):
        neighbor = state.copy()
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    return neighbors

def hill_climbing():
    current = [0]*len(questions)
    current_cost, _, _ = evaluate(current)

    improved = True
    while improved:
        improved = False
        neighbors = get_neighbors(current)
        for neighbor in neighbors:
            cost, _, _ = evaluate(neighbor)
            if cost < current_cost:
                current = neighbor
                current_cost = cost
                improved = True
                break
    return current, current_cost

solution, cost = hill_climbing()
time_sum = sum(q["Time_min"] for q,s in zip(questions,solution) if s==1)
diff_sum = sum(q["Difficulty"] for q,s in zip(questions,solution) if s==1)
selected_questions = [q["QuestionID"] for q,s in zip(questions,solution) if s==1]

print('================================================')
print("Costo final:", cost)
print('================================================')
print("Tiempo total:", time_sum)
print('================================================')
print("Dificultad total:", diff_sum)
print('================================================')
print("Preguntas seleccionadas:", selected_questions)
print('================================================')
