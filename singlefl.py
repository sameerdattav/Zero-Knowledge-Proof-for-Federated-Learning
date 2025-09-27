import numpy as np
import hashlib

# --- Manual Neural Network Implementation ---
# We are building a simple two-layer neural network from scratch
# to demonstrate the gradient descent process without high-level libraries.

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return x * (1 - x)

class SimpleNN:
    """A simple two-layer neural network."""
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
        # Initialize biases with zeros
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def get_weights(self):
        """Returns the current weights and biases of the model."""
        return (self.weights_input_hidden, self.weights_hidden_output, self.bias_hidden, self.bias_output)

    def set_weights(self, weights):
        """Sets the weights and biases of the model."""
        self.weights_input_hidden, self.weights_hidden_output, self.bias_hidden, self.bias_output = weights

    def forward(self, inputs):
        """Performs the forward pass."""
        # Hidden layer
        self.hidden_sum = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.activated_hidden = sigmoid(self.hidden_sum)
        # Output layer
        self.output_sum = np.dot(self.activated_hidden, self.weights_hidden_output) + self.bias_output
        self.activated_output = sigmoid(self.output_sum)
        return self.activated_output

    def backward(self, inputs, targets, learning_rate):
        """
        Performs the backward pass and updates weights (gradient descent).
        This is the core of the training process.
        """
        # Calculate the error
        error = targets - self.activated_output
        
        # Calculate gradients for the output layer
        d_output = error * sigmoid_derivative(self.activated_output)
        
        # Calculate error for the hidden layer
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        
        # Calculate gradients for the hidden layer
        d_hidden = error_hidden * sigmoid_derivative(self.activated_hidden)
        
        # --- Gradient Calculation ---
        grad_weights_hidden_output = self.activated_hidden.T.dot(d_output)
        grad_bias_output = np.sum(d_output, axis=0, keepdims=True)
        grad_weights_input_hidden = inputs.T.dot(d_hidden)
        grad_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True)

        # --- Weight Update (Gradient Descent Step) ---
        self.weights_hidden_output += grad_weights_hidden_output * learning_rate
        self.bias_output += grad_bias_output * learning_rate
        self.weights_input_hidden += grad_weights_input_hidden * learning_rate
        self.bias_hidden += grad_bias_hidden * learning_rate
        
        # Return the gradients (in a real ZKP scenario, you'd prove knowledge of these)
        return (grad_weights_input_hidden, grad_weights_hidden_output, grad_bias_hidden, grad_bias_output)

# --- Merkle Tree for Data Verification ---
# A simple Merkle Tree to create a root hash representing the client's dataset.
# This ensures data integrity and privacy without revealing the data itself.

def hash_data(data):
    """Hashes a single data point."""
    return hashlib.sha256(str(data).encode()).hexdigest()

def build_merkle_tree(data_list):
    """Builds a Merkle Tree and returns the root hash."""
    if not data_list:
        return None
    
    # Hash each individual data point
    hashed_leaves = [hash_data(d) for d in data_list]
    
    # Iteratively hash pairs of nodes until we have one root
    while len(hashed_leaves) > 1:
        if len(hashed_leaves) % 2 != 0:
            hashed_leaves.append(hashed_leaves[-1]) # Duplicate last element if odd
        
        new_level = []
        for i in range(0, len(hashed_leaves), 2):
            combined_hash = hash_data(hashed_leaves[i] + hashed_leaves[i+1])
            new_level.append(combined_hash)
        hashed_leaves = new_level
        
    return hashed_leaves[0]

# --- Federated Learning Components ---

class Client:
    """Represents a client in the federated learning system."""
    def __init__(self, client_id, model, X_train, y_train):
        self.client_id = client_id
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        # The Merkle root represents the client's private dataset
        self.dataset_merkle_root = build_merkle_tree(X_train.tolist())

    def train(self, epochs=1, learning_rate=0.1):
        """Trains the client's model on its local data."""
        print(f"Client {self.client_id}: Starting training...")
        print(f"  - Dataset Merkle Root: {self.dataset_merkle_root}")
        for epoch in range(epochs):
            # Forward pass
            self.model.forward(self.X_train)
            # Backward pass (Gradient Descent)
            self.model.backward(self.X_train, self.y_train, learning_rate)
        print(f"Client {self.client_id}: Training complete.")
        return self.model.get_weights()

class Server:
    """Represents the central server."""
    def __init__(self, initial_model):
        self.global_model = initial_model

    def aggregate_weights(self, client_weights_list):
        """Averages the weights from all clients (Federated Averaging)."""
        # Unzip the weights from all clients
        w_ih, w_ho, b_h, b_o = zip(*client_weights_list)
        
        # Average the weights and biases
        avg_w_ih = np.mean(w_ih, axis=0)
        avg_w_ho = np.mean(w_ho, axis=0)
        avg_b_h = np.mean(b_h, axis=0)
        avg_b_o = np.mean(b_o, axis=0)

        aggregated_weights = (avg_w_ih, avg_w_ho, avg_b_h, avg_b_o)
        self.global_model.set_weights(aggregated_weights)
        print("\nServer: Aggregated client weights to update the global model.\n")

    def distribute_model(self, clients):
        """Sends the current global model weights to all clients."""
        global_weights = self.global_model.get_weights()
        for client in clients:
            client.model.set_weights(global_weights)
        print("Server: Distributed global model to all clients.")

# --- Simulation Setup ---

def create_private_datasets(num_clients):
    """Generates distinct, private datasets for each client."""
    datasets = []
    # XOR problem dataset base
    base_X = np.array([[0,0], [0,1], [1,0], [1,1]])
    base_y = np.array([[0], [1], [1], [0]])
    
    for i in range(num_clients):
        # Create slightly different data for each client
        X = base_X + np.random.normal(0, 0.1 * (i + 1), base_X.shape)
        y = base_y
        datasets.append((X, y))
        print(f"Generated private dataset for Client {i+1}")
    return datasets

# --- Main Simulation ---

if __name__ == "__main__":
    # Configuration
    NUM_CLIENTS = 4
    INPUT_SIZE = 2
    HIDDEN_SIZE = 5
    OUTPUT_SIZE = 1
    COMMUNICATION_ROUNDS = 5
    LOCAL_EPOCHS = 200

    # 1. Initialize the global model and the server
    global_nn = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    server = Server(global_nn)

    # 2. Create private datasets
    client_datasets = create_private_datasets(NUM_CLIENTS)

    # 3. Initialize clients with their private data
    clients = []
    for i in range(NUM_CLIENTS):
        client_model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        X, y = client_datasets[i]
        client = Client(client_id=i+1, model=client_model, X_train=X, y_train=y)
        clients.append(client)
    
    print("\n--- Federated Learning Simulation Initialized ---")

    # 4. Run federated learning rounds
    for round_num in range(COMMUNICATION_ROUNDS):
        print(f"--- Starting Communication Round {round_num + 1}/{COMMUNICATION_ROUNDS} ---")
        
        # a. Server distributes the global model to all clients
        server.distribute_model(clients)
        
        # b. Clients train the model on their private data
        client_weights = []
        for client in clients:
            weights = client.train(epochs=LOCAL_EPOCHS)
            client_weights.append(weights)
            
        # c. Server aggregates the weights from clients
        server.aggregate_weights(client_weights)

    print("--- Federated Learning Simulation Finished ---\n")
    
    # 5. Evaluate the final global model
    final_global_model = server.global_model
    test_data = np.array([[0,0], [0,1], [1,0], [1,1]])
    predictions = final_global_model.forward(test_data)
    
    print("Final Global Model Predictions (XOR Problem):")
    for i, data in enumerate(test_data):
        # Round the raw sigmoid output (which is between 0 and 1) to a final class
        final_answer = 1 if predictions[i][0] > 0.5 else 0
        print(f"xor({data[0]}, {data[1]}) = {final_answer} (Raw Prediction: {predictions[i][0]:.4f})")