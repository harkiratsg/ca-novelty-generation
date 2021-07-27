""" Cellular Automata Genetic Algorithm """

import torch
import random
import math
from utils import *

def generate_random_population(size, resolution):
    population = []
    for i in range(size):
        rule = get_random_continious_rule(resolution)
        population.append([rule, 0])
    
    return population

def compute_rule_fitness(rule, width, generation_steps):
    generated = generate_continuous_rule_states(rule, np.random.rand(width), generation_steps)
    return total_sparsity(generated, 4)

def get_population_fitness_list(population):
    return [p[1] for p in population]
    
def compute_population_fitness(population, width, generation_steps, fitness_sample_size):
    # Compute fitness for each rule
    for i, rule_fitness in enumerate(population):
        # Compute average fitness of rule
        fitness = 0
        for s in range(fitness_sample_size):
            fitness += compute_rule_fitness(rule_fitness[0], width, generation_steps) / fitness_sample_size
        
        population[i][1] = fitness
    
    return population

def select(population, pressure):
    # Sort based on fitness
    population.sort(key=lambda x: x[1])
    
    # Remove rules with the lowest fitness based on pressure
    population = population[int(pressure * len(population)):]
    
    # Return remaining population
    return population

def generate_rule_offspring(rule, mutation_rate, mutation_degree):
    return mutate(rule.copy(), mutation_rate, mutation_degree)
    
def generate_population_offspring(population, max_offspring, mutation_rate, mutation_degree):
    offspring = []
    
    # Keep generating offspring until we hit the max offspring number
    while len(offspring) < max_offspring:
        # Shuffle rules and generate children for 
        random.shuffle(population)
        
        for rule, fitness in population:
            offspring.append([generate_rule_offspring(rule, mutation_rate, mutation_degree), 0])
            
            # Break if max offspring reached
            if len(offspring) > max_offspring:
                break
    
    return offspring
    
def mutate(rule, mutation_rate, mutation_degree):
    for i in range(rule.shape[0]):
        for j in range(rule.shape[1]):
            for k in range(rule.shape[2]):
                if random.random() < mutation_rate:
                    if rule[i][j][k] >= 0.5:
                        rule[i][j][k] -= rule[i][j][k] * random.random() * mutation_degree
                    elif rule[i][j][k] < 0.5:
                        rule[i][j][k] += rule[i][j][k] * random.random() * mutation_degree
    return rule

        
        
def run_generations(num_generations, resolution, initial_population, width, selection_pressure, mutation_rate, mutation_degree, generation_steps):
    
    min_fitness_values = []
    average_fitness_values = []
    max_fitness_values = []
    
    # Initialize population
    population = generate_random_population(size=initial_population, resolution=resolution)
    
    # Compute fitness for entire population
    population = compute_population_fitness(population, width=width, generation_steps=generation_steps, fitness_sample_size=5)
    
    # Run generations
    for g in range(num_generations):
        print("poop")
        # Select strongest from population
        population = select(population, pressure=selection_pressure)
        
        # Generate offspring
        population += generate_population_offspring(population, initial_population - len(population), mutation_rate, mutation_degree)
        
        # Sort and save fitness stats
        population = compute_population_fitness(population, width=width, generation_steps=generation_steps, fitness_sample_size=5)
        
        population.sort(key=lambda x: x[1])
        
        fitness_values = get_population_fitness_list(population)
        min_fitness_values.append(min(fitness_values))
        average_fitness_values.append(sum(fitness_values) / len(population))
        max_fitness_values.append(max(fitness_values))
        
        print("Generation ", g)
        print("Population {} Fitness: {}\n".format(len(population), average_fitness_values[-1]))
    
    # Select strongest from final population
    population = compute_population_fitness(population, width=width, generation_steps=100, fitness_sample_size=5)
    population = select(population, pressure=selection_pressure)
    
    return population, min_fitness_values, average_fitness_values, max_fitness_values


if __name__ == "__main__":
    
    WIDTH = 10
    
    NUM_GEN = 0
    RES = 3
    INITIAL_POP = 300
    PRESSURE = 0.4
    MUT_RATE = 0.01
    MUT_DEGREE = 0.1
    GEN_STEPS = 100
    
    EVAL_SAMPLES = 2
    EVAL_GEN_STEPS = 100
    
    PATH = "generated/pics/ga_entropy/"
    
    population, min_fitness_values, average_fitness_values, max_fitness_values = run_generations(
            num_generations=NUM_GEN,
            resolution=RES,
            initial_population=INITIAL_POP,
            width=WIDTH,
            selection_pressure=PRESSURE,
            mutation_rate=MUT_RATE,
            mutation_degree=MUT_DEGREE,
            generation_steps=GEN_STEPS
        )
    
    # Plot fitness
    #plt.plot(min_fitness_values)
    #plt.plot(average_fitness_values)
    #plt.plot(max_fitness_values)
    #plt.savefig("fitness.png")
    #plt.show()
    
    print("Training")
    fitness_values = get_population_fitness_list(population)
    print(f"Gens: {NUM_GEN}, Res {RES}, Pop: {INITIAL_POP}, Pressure: {PRESSURE}, Mut Rate: {MUT_RATE}, Mut Degree: {MUT_DEGREE}, Gen Steps: {GEN_STEPS}")
    print("Mean:", np.mean(np.array(fitness_values)))
    print("Std :", np.std(np.array(fitness_values)))
    print("Max :", max(fitness_values))
    print("Min :", min(fitness_values))
    
    # Evaluation
    evaluate([p[0] for p in population], PATH, WIDTH, EVAL_SAMPLES, EVAL_GEN_STEPS)
