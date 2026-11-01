import numpy as np
import random
POPULATION_SIZE, GENES, GENERATIONS = 50, 5, 30
MUTATION_RATE, CROSSOVER_RATE = 0.1, 0.7
def fitness(solution):
    return np.sum(solution ** 2)
def select(population, scores):
    candidates = random.sample(range(len(population)), 3)
    return population[min(candidates, key=lambda i: scores[i])].copy()
def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENES - 1)
        return np.concatenate((p1[:point], p2[point:])), np.concatenate((p2[:point], p1[point:]))
    return p1.copy(), p2.copy()
def mutate(child):
    for j in range(GENES):
        if random.random() < MUTATION_RATE:
            child[j] += np.random.uniform(-1, 1)
    return child
population = np.random.uniform(-10, 10, (POPULATION_SIZE, GENES))
best_solution, best_score = None, float('inf')
for generation in range(GENERATIONS):
    scores = np.array([fitness(x) for x in population])
    index = np.argmin(scores)
    if scores[index] < best_score:
        best_solution, best_score = population[index].copy(), scores[index]
    parents = [select(population, scores) for _ in range(POPULATION_SIZE)]
    offspring = []
    for i in range(0, POPULATION_SIZE, 2):
        c1, c2 = crossover(parents[i], parents[(i + 1) % POPULATION_SIZE])
        offspring.extend([mutate(c1), mutate(c2)])
    population = np.array(offspring[:POPULATION_SIZE])
    print(f"Generation {generation + 1}: Best Fitness = {best_score}")
print("Best Solution:", best_solution)
print("Best Fitness:", best_score)