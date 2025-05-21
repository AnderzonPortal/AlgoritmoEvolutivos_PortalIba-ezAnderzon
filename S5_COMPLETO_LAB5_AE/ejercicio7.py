import pandas as pd
import numpy as np
import random
from collections import Counter

# Leer CSV con separador ';'
df = pd.read_csv('E:\\CICLO IX\\ALGORITMOS EVOLUTIVOS\\S5_AVANCE_LAB5_AE\\bd\\estudiantes.csv', sep=';')

students = df.to_dict('records')  # lista de dicts con keys: 'StudentID', 'GPA', 'Skill'

NUM_TEAMS = 5
TEAM_SIZE = 4

skills_list = [s["Skill"] for s in students]
unique_skills = set(skills_list)

# Distribución ideal de skills por equipo
total_skill_counts = Counter(skills_list)
ideal_skill_per_team = {k: v / NUM_TEAMS for k, v in total_skill_counts.items()}

def fitness(solution):
    gpa_vars = 0
    skill_penalty = 0

    for team in solution:
        gpas = [students[i]["GPA"] for i in team]
        gpa_vars += np.var(gpas)

        team_skills = [students[i]["Skill"] for i in team]
        team_counts = Counter(team_skills)
        for skill in unique_skills:
            expected = ideal_skill_per_team.get(skill, 0)
            actual = team_counts.get(skill, 0)
            skill_penalty += abs(actual - expected)

    return gpa_vars + skill_penalty

def get_neighbors(solution):
    neighbors = []
    for i in range(NUM_TEAMS):
        for j in range(i + 1, NUM_TEAMS):
            for a in range(TEAM_SIZE):
                for b in range(TEAM_SIZE):
                    neighbor = [team.copy() for team in solution]
                    neighbor[i][a], neighbor[j][b] = neighbor[j][b], neighbor[i][a]
                    neighbors.append(neighbor)
    return neighbors

def random_solution():
    indices = list(range(NUM_TEAMS * TEAM_SIZE))
    random.shuffle(indices)
    return [indices[i * TEAM_SIZE:(i + 1) * TEAM_SIZE] for i in range(NUM_TEAMS)]

def hill_climbing(max_iter=1000):
    current = random_solution()
    current_fit = fitness(current)

    for _ in range(max_iter):
        neighbors = get_neighbors(current)
        improved = False
        for neighbor in neighbors:
            fit = fitness(neighbor)
            if fit < current_fit:
                current = neighbor
                current_fit = fit
                improved = True
                break
        if not improved:
            break

    return current, current_fit

solution, fit = hill_climbing()

print('================================================')
print(f"Fitness final: {fit:.4f}\n")
for i, team in enumerate(solution):
    print('================================================')
    print(f"Equipo {i+1}:")
    for idx in team:
        s = students[idx]
        print(f"  {s['StudentID']} - GPA: {s['GPA']:.2f} - Skill: {s['Skill']}")
    print()
print('================================================')