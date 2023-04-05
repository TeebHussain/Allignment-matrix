import numpy as np
import random
import pandas as pd

data = pd.read_csv(r"C:\Users\Shaik Teeb Hussain\Downloads\archive (8)\Bank_Personal_Loan_Modelling.csv")
# Define the neural network
class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # Initialize the weights of the neural network
        self.input_weights = np.random.rand(num_inputs, num_hidden)
        self.output_weights = np.random.rand(num_hidden, num_outputs)
        
    def predict(self, inputs):
        # Compute the outputs of the neural network given the inputs
        hidden_outputs = np.dot(inputs, self.input_weights)
        hidden_outputs = 1 / (1 + np.exp(-hidden_outputs))
        outputs = np.dot(hidden_outputs, self.output_weights)
        outputs = 1 / (1 + np.exp(-outputs))
        return outputs

# Define the Particle class for PSO
class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.zeros_like(position)
        self.best_position = position
        self.best_score = float('inf')
        
    def update_velocity(self, global_best_position, c1, c2):
        r1 = random.random()
        r2 = random.random()
        self.velocity = self.velocity + c1 * r1 * (self.best_position - self.position) + c2 * r2 * (global_best_position - self.position)
        
    def update_position(self):
        self.position = self.position + self.velocity
        
    def evaluate(self, neural_network, x_train, y_train):
        predicted_y = np.zeros_like(y_train)
        for i in range(len(x_train)):
            predicted_y[i] = neural_network.predict(x_train[i])
        score = np.mean(np.square(predicted_y - y_train))
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position
        return score
        
# Define the PSO function
def pso(neural_network, x_train, y_train, num_particles, num_iterations, c1, c2):
    # Initialize the particles
    particles = [Particle(neural_network.output_weights.flatten()) for i in range(num_particles)]
    
    # Find the global best particle
    global_best_particle = particles[0]
    for particle in particles:
        if particle.best_score < global_best_particle.best_score:
            global_best_particle = particle
            
    # Train the neural network using PSO
    for iteration in range(num_iterations):
        for particle in particles:
            particle.update_velocity(global_best_particle.best_position, c1, c2)
            particle.update_position()
            particle.evaluate(neural_network, x_train, y_train)
            if particle.best_score < global_best_particle.best_score:
                global_best_particle = particle
                
    # Update the weights of the neural network using the global best particle
    neural_network.output_weights = np.reshape(global_best_particle.best_position, neural_network.output_weights.shape)
    
    return neural_network

# Example usage
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])
neural_network = NeuralNetwork(2, 2, 1)
neural_network = pso(neural_network, x_train, y_train, num_particles=10, num_iterations=100, c1=2, c2=2)
print(neural_network.predict(x_train))
