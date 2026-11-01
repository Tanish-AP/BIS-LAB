import numpy as np
def optimization_function(position):
    return sum(x**2 for x in position)
GRID_SIZE = (10, 10)
NEIGHBORHOOD_RADIUS = 1
DIMENSIONS = 2
ITERATIONS = 30
def initialize_population(grid_size, dimensions):
    population = np.random.uniform(-10, 10, size=(grid_size[0], grid_size[1], dimensions))
    return population
def evaluate_fitness(population):
    fitness = np.zeros((population.shape[0], population.shape[1]))
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            fitness[i, j] = optimization_function(population[i, j])
    return fitness
def get_neighborhood(grid, x, y, radius):
    neighbors = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i == 0 and j == 0:
                continue
            ni, nj = x + i, y + j
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                neighbors.append((ni, nj))
    return neighbors
def update_states(population, fitness):
    new_population = np.copy(population)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            neighbors = get_neighborhood(population, i, j, NEIGHBORHOOD_RADIUS)
            best_neighbor = population[i, j]
            best_fitness = fitness[i, j]
            for ni, nj in neighbors:
                if fitness[ni, nj] < best_fitness:
                    best_fitness = fitness[ni, nj]
                    best_neighbor = population[ni, nj]
            new_population[i, j] = (population[i, j] + best_neighbor) / 2
    return new_population
def parallel_cellular_algorithm():
    population = initialize_population(GRID_SIZE, DIMENSIONS)
    best_solution = None
    best_fitness = float('inf')
    for iteration in range(ITERATIONS):
        fitness = evaluate_fitness(population)
        min_fitness = np.min(fitness)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[np.unravel_index(np.argmin(fitness), fitness.shape)]
        population = update_states(population, fitness)
        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")
    print("\nBest Solution Found:")
    print(f"Position: {best_solution}, Fitness: {best_fitness}")
if __name__ == "__main__":
    parallel_cellular_algorithm()