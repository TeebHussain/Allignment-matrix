import numpy as np
import pandas as pd

data = pd.read_csv(r"C:\Users\Shaik Teeb Hussain\Downloads\archive (8)\Bank_Personal_Loan_Modelling.csv")

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the neural network class
class NeuralNetworkACO:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights and biases with random values
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.b1 = np.random.randn(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim)
        self.b2 = np.random.randn(self.output_dim)

    # Define the forward propagation function
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        y_hat = sigmoid(self.z2)
        return y_hat

    # Define the loss function
    def loss(self, X, y):
        y_hat = self.forward(X)
        return np.sum((y - y_hat) ** 2)

    # Define the ACO algorithm
    def aco(self, X, y, n_ants, alpha, beta, rho, q):
        # Initialize pheromone matrix
        tau = np.ones((self.hidden_dim, self.output_dim))

        # Initialize best weights and biases
        best_W1 = self.W1.copy()
        best_b1 = self.b1.copy()
        best_W2 = self.W2.copy()
        best_b2 = self.b2.copy()
        best_loss = np.inf

        for i in range(n_ants):
            # Create new weights and biases based on pheromone levels
            W1_new = np.zeros_like(self.W1)
            b1_new = np.zeros_like(self.b1)
            W2_new = np.zeros_like(self.W2)
            b2_new = np.zeros_like(self.b2)

            for j in range(self.hidden_dim):
                for k in range(self.output_dim):
                    # Calculate the probability of selecting each weight
                    p = tau[j, k] ** alpha / (np.sum(tau ** alpha))

                    # Select a weight based on the probability
                    if np.random.uniform() < p:
                        W1_new[:, j] += self.W1[:, j] + beta * np.random.randn(self.input_dim)
                        W2_new[j, k] += self.W2[j, k] + beta * np.random.randn()
                        b1_new[j] += self.b1[j] + beta * np.random.randn()
                        b2_new[k] += self.b2[k] + beta * np.random.randn()

            # Evaluate the new weights and biases
            nn = NeuralNetworkACO(self.input_dim, self.hidden_dim, self.output_dim)
            nn.W1 = W1_new
            nn.b1 = b1_new
            nn.W2 = W2_new
            nn.b2 = b2_new
            loss = nn.loss(X, y)

            # Update the pheromone matrix
            tau += rho * (1 / loss) * (W1_new * W2_new)
            
            # Update the best weights and biases
            if loss < best_loss:
                best_W1 = W1_new.copy()
                best_b1 = b1_new.copy()
                best_W2 = W2_new.copy
