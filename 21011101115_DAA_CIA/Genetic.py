import numpy as np
import random
import pandas as pd

data = pd.read_csv(r"C:\Users\Shaik Teeb Hussain\Downloads\archive (8)\Bank_Personal_Loan_Modelling.csv")

# Define neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)
    
    def forward(self, inputs):
        hidden_layer = np.dot(inputs, self.weights1)
        activated_hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(activated_hidden_layer, self.weights2)
        activated_output_layer = self.sigmoid(output_layer)
        return activated_output_layer
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Define genetic algorithm
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, neural_network):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.neural_network = neural_network
        self.population = []
        for i in range(population_size):
            individual = {}
            individual['weights1'] = np.random.rand(neural_network.weights1.shape[0], neural_network.weights1.shape[1])
            individual['weights2'] = np.random.rand(neural_network.weights2.shape[0], neural_network.weights2.shape[1])
            self.population.append(individual)
    
    def selection(self):
        fitness_scores = []
        for individual in self.population:
            fitness_scores.append(self.calculate_fitness(individual))
        sorted_indices = np.argsort(fitness_scores)
        sorted_population = [self.population[i] for i in sorted_indices]
        return sorted_population[:int(self.population_size / 2)]
    
    def crossover(self, selected_population):
        new_population = []
        for i in range(self.population_size - len(selected_population)):
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            child = {}
            child['weights1'] = np.zeros_like(parent1['weights1'])
            child['weights2'] = np.zeros_like(parent1['weights2'])
            for j in range(parent1['weights1'].shape[0]):
                for k in range(parent1['weights1'].shape[1]):
                    if random.random() < 0.5:
                        child['weights1'][j][k] = parent1['weights1'][j][k]
                    else:
                        child['weights1'][j][k] = parent2['weights1'][j][k]
            for j in range(parent1['weights2'].shape[0]):
                for k in range(parent1['weights2'].shape[1]):
                    if random.random() < 0.5:
                        child['weights2'][j][k] = parent1['weights2'][j][k]
                    else:
                        child['weights2'][j][k] = parent2['weights2'][j][k]
            new_population.append(child)
        return new_population
    
    def mutation(self, new_population):
        for i in range(len(new_population)):
            for j in range(new_population[i]['weights1'].shape[0]):
                for k in range(new_population[i]['weights1'].shape[1]):
                    if random.random() < self.mutation_rate:
                        new_population[i]['weights1'][j][k] = random.random()
            for j in range(new_population[i]['weights2'].shape[0]):
                for k in range(new_population[i]['weights2'].shape[1]):
                    if random.random() < self.mutation_rate:
                        new_population[i]['weights2'][j][k] = random.random()
        return new_population
    
    def calculate_fitness(self)
