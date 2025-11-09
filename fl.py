import torch  # <--- PYTORCH: Replaces numpy for tensor operations
import torch.nn as nn  # <--- PYTORCH: For building neural network layers
import torch.optim as optim  # <--- PYTORCH: For the optimizer
import numpy as np  # <--- Still used for initial data setup
import hashlib

# --- PyTorch Neural Network Implementation ---
# We convert the SimpleNN to a torch.nn.Module.
# PyTorch will handle the forward/backward graph and gradients automatically.

# <--- DELETED: sigmoid and sigmoid_derivative functions are now built-in.

class PyTorchNN(nn.Module):  # <--- PYTORCH: Inherit from nn.Module
    """A simple two-layer neural network in PyTorch."""
    def __init__(self, input_size, hidden_size, output_size):
        super(PyTorchNN, self).__init__()  # <--- PYTORCH: Call parent constructor
        # <--- PYTORCH: Define layers. Weights and biases are managed internally.
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    # <--- DELETED: get_weights and set_weights are replaced by
    # state_dict() and load_state_dict()

    def forward(self, inputs):
        """Performs the forward pass."""
        # <--- PYTORCH: Simpler forward pass syntax
        x = self.layer1(inputs)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

    # <--- DELETED: The entire 'backward' function is now handled
    # automatically by PyTorch's autograd (e.g., loss.backward())

# --- Merkle Tree for Data Verification ---
# This code is identical, as it doesn't depend on Numpy or PyTorch.

def hash_data(data):
    """Hashes a single data point."""
    return hashlib.sha256(str(data).encode()).hexdigest()

def build_merkle_tree(data_list):
    """Builds a Merkle Tree and returns the root hash."""
    if not data_list:
        return None
    
    hashed_leaves = [hash_data(d) for d in data_list]
    
    while len(hashed_leaves) > 1:
        if len(hashed_leaves) % 2 != 0:
            hashed_leaves.append(hashed_leaves[-1]) 
        
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
        self.dataset_merkle_root = build_merkle_tree(X_train.tolist())

    def train(self, epochs=1, learning_rate=0.1):
        """Trains the client's model on its local data using PyTorch."""
        print(f"Client {self.client_id}: Starting training...")
        print(f"  - Dataset Merkle Root: {self.dataset_merkle_root}")

        # <--- PYTORCH: Set up the optimizer and loss function
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()  # Mean Squared Error Loss

        for epoch in range(epochs):
            # <--- PYTORCH: Standard PyTorch training loop
            optimizer.zero_grad()  # Clear previous gradients
            outputs = self.model(self.X_train)  # Forward pass
            loss = criterion(outputs, self.y_train)  # Calculate loss
            loss.backward()  # Automatic backward pass (calculates gradients)
            optimizer.step()  # Update weights (replaces your manual SGD)

        print(f"Client {self.client_id}: Training complete.")
        # <--- PYTORCH: Return the model's weights as a state dictionary
        return self.model.state_dict()

class Server:
    """Represents the central server orchestrating the FL process."""
    def __init__(self, initial_model):
        self.global_model = initial_model

    def aggregate_weights(self, client_state_dicts):
        """Averages the weights (state_dicts) from all clients."""
        # <--- PYTORCH: This is the logic for Federated Averaging with state_dicts
        avg_state_dict = self.global_model.state_dict()
        
        for key in avg_state_dict.keys():
            # Stack all client tensors for the current key
            client_tensors = torch.stack([state_dict[key] for state_dict in client_state_dicts])
            # Compute the average
            avg_tensor = torch.mean(client_tensors, dim=0)
            # Update the average state_dict
            avg_state_dict[key] = avg_tensor

        self.global_model.load_state_dict(avg_state_dict)
        print("\nServer: Aggregated client weights to update the global model.\n")

    def distribute_model(self, clients):
        """Sends the current global model state_dict to all clients."""
        global_state_dict = self.global_model.state_dict()
        for client in clients:
            # <--- PYTORCH: Load the state_dict into the client's model
            client.model.load_state_dict(global_state_dict)
        print("Server: Distributed global model to all clients.")

# --- Simulation Setup ---

def create_private_datasets(num_clients):
    """Generates distinct, private datasets for each client as PyTorch Tensors."""
    datasets = []
    base_X = np.array([[0,0], [0,1], [1,0], [1,1]])
    base_y = np.array([[0], [1], [1], [0]])
    
    for i in range(num_clients):
        X_np = base_X + np.random.normal(0, 0.1 * (i + 1), base_X.shape)
        y_np = base_y
        
        # <--- PYTORCH: Convert data to Tensors
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)
        
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
    LOCAL_EPOCHS = 2000

    # 1. Initialize the global model and the server
    global_nn = PyTorchNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)  # <--- Use new class
    server = Server(global_nn)

    # 2. Create private datasets for each client
    client_datasets = create_private_datasets(NUM_CLIENTS)

    # 3. Initialize clients
    clients = []
    for i in range(NUM_CLIENTS):
        client_model = PyTorchNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) # <--- Use new class
        X, y = client_datasets[i]
        client = Client(client_id=i+1, model=client_model, X_train=X, y_train=y)
        clients.append(client)
    
    print("\n--- Federated Learning Simulation Initialized ---")

    # 4. Run federated learning rounds
    for round_num in range(COMMUNICATION_ROUNDS):
        print(f"--- Starting Communication Round {round_num + 1}/{COMMUNICATION_ROUNDS} ---")
        
        server.distribute_model(clients)
        
        client_weights = []
        for client in clients:
            weights = client.train(epochs=LOCAL_EPOCHS, learning_rate=0.1)
            client_weights.append(weights)
            
        server.aggregate_weights(client_weights)

    print("--- Federated Learning Simulation Finished ---\n")
    
    # 5. Evaluate the final global model
    final_global_model = server.global_model
    test_data_np = np.array([[0,0], [0,1], [1,0], [1,1]])
    expected_output = np.array([[0], [1], [1], [0]])
    
    # <--- PYTORCH: Convert test data for model
    test_data_tensor = torch.tensor(test_data_np, dtype=torch.float32)

    # <--- PYTORCH: Get predictions in no_grad context
    final_global_model.eval() # Set model to evaluation mode
    with torch.no_grad():
        predictions_tensor = final_global_model(test_data_tensor)
    
    # <--- PYTORCH: Convert predictions back to numpy for printing
    predictions = predictions_tensor.numpy()
    
    print("--- Final Model Evaluation ---")
    print("Evaluating the global model on the XOR test data:")
    
    for i in range(len(test_data_np)):
        input_val = test_data_np[i]
        prediction_val = predictions[i][0]
        expected_val = expected_output[i][0]
        predicted_class = 1 if prediction_val > 0.5 else 0
        
        print(
            f"Input: {input_val} -> "
            f"Expected: {expected_val} | "
            f"Raw Prediction: {prediction_val:.4f} -> "
            f"Final Prediction: {predicted_class}"
        )