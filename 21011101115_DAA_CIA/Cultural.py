import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random

import pandas as pd

data = pd.read_csv(r"C:\Users\Shaik Teeb Hussain\Downloads\archive (8)\Bank_Personal_Loan_Modelling.csv")

# Load dataset
digits = load_digits()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network architecture
input_size = X_train.shape[1]
hidden_size = 100
output_size = len(np.unique(y_train))

# Define the initial population size
population_size = 10

# Define the number of iterations to run the cultural algorithm
num_iterations = 100

# Define the mutation rate
mutation_rate = 0.1

# Initialize the population
population = []
for i in range(population_size):
    # Generate random weights for the neural network
    weights1 = np.random.randn(input_size, hidden_size)
    weights2 = np.random.randn(hidden_size, output_size)
    population.append((weights1, weights2))

# Define the fitness function
def fitness_function(weights):
    # Calculate the output of the neural network
    hidden_layer = np.maximum(0, np.dot(X_train, weights[0]))
    output_layer = np.dot(hidden_layer, weights[1])
    
    # Calculate the accuracy of the neural network
    y_pred = np.argmax(output_layer, axis=1)
    accuracy = accuracy_score(y_train, y_pred)
    
    return accuracy

# Run the cultural algorithm
for iteration in range(num_iterations):
    # Evaluate the fitness of each individual in the population
    fitness_scores = [fitness_function(individual) for individual in population]
    
    # Sort the population by fitness score
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    
    # Define the knowledge source
    knowledge_source = sorted_population[0]
    
    # Update the population
    new_population = []
    for i in range(population_size):
        # Select an individual from the population
        individual = random.choice(sorted_population)
        
        # Mutate the individual
        weights1 = individual[0] + mutation_rate * np.random.randn(input_size, hidden_size)
        weights2 = individual[1] + mutation_rate * np.random.randn(hidden_size, output_size)
        new_individual = (weights1, weights2)
        
        # Combine the individual with the knowledge source
        combined_individual = (knowledge_source[0] + new_individual[0], knowledge_source[1] + new_individual[1])
        
        # Add the combined individual to the new population
        new_population.append(combined_individual)
        
    population = new_population
    
# Evaluate the fitness of the final population
fitness_scores = [fitness_function(individual) for individual in population]

# Sort the population by fitness score
sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

# Get the best individual from the population
best_individual = sorted_population[0]

# Calculate the output of the neural network on the testing set
hidden_layer = np.maximum(0, np.dot(X_test, best_individual[0]))
output_layer = np.dot(hidden_layer, best_individual[1])
y_pred = np.argmax(output_layer, axis=1)

# Calculate the accuracy of the neural network on the
