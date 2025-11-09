# import torch
# import torch.onnx
# import torch.nn as nn

# # Import the model class *only* to get its parameter shapes
# try:
#     from fl import PyTorchNN
# except ImportError:
#     print("Error: Make sure 'pytorch_fl.py' is in the same directory.")
#     # Define it here just in case, so the script can run
#     class PyTorchNN(nn.Module):
#         def __init__(self, input_size, hidden_size, output_size):
#             super(PyTorchNN, self).__init__()
#             self.layer1 = nn.Linear(input_size, hidden_size)
#             self.layer2 = nn.Linear(hidden_size, output_size)
#             self.sigmoid = nn.Sigmoid()
#         def forward(self, inputs):
#             x = self.layer1(inputs)
#             x = self.sigmoid(x)
#             x = self.layer2(x)
#             x = self.sigmoid(x)
#             return x

# # --- Define Model Constants ---
# INPUT_SIZE = 2
# HIDDEN_SIZE = 5
# OUTPUT_SIZE = 1
# LEARNING_RATE = 0.1

# # --- Create a "Wrapper" Module for Export ---
# #
# # This class is an nn.Module that functionally implements the
# # *entire* training step (fwd, loss, back, update) using
# # pure tensor ops, just like your numpy script.
# # This is what ONNX can trace.
# #
# class FullTrainingStep(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # We just need the learning rate as a constant
#         self.lr = torch.tensor(LEARNING_RATE)
#         # We need the sigmoid function
#         self.sigmoid = nn.Sigmoid()

#     def sigmoid_derivative(self, sigmoid_output):
#         # Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
#         return sigmoid_output * (1 - sigmoid_output)

#     def forward(self, data, labels, l1_w, l1_b, l2_w, l2_b):
        
#         # --- 1. Forward Pass ---
#         # Re-implementing the model's forward pass functionally
        
#         # Hidden layer
#         # Note: PyTorch nn.Linear does (batch, in) @ (out, in).T
#         # So functional matmul is (batch, in) @ (in, out)
#         hidden_sum = torch.matmul(data, l1_w.t()) + l1_b
#         activated_hidden = self.sigmoid(hidden_sum)
        
#         # Output layer
#         output_sum = torch.matmul(activated_hidden, l2_w.t()) + l2_b
#         activated_output = self.sigmoid(output_sum)
        
#         # --- 2. Loss Calculation (MSE) ---
#         # We just need the error for the backward pass
#         # error = targets - activated_output
#         error = labels - activated_output
        
#         # --- 3. Backward Pass (Manual Gradient Calculation) ---
        
#         # Gradients for the output layer (d_output from your numpy script)
#         d_output = error * self.sigmoid_derivative(activated_output)
        
#         # Error for the hidden layer (error_hidden from your numpy script)
#         # (batch, 1) @ (1, 5) -> (batch, 5)
#         error_hidden = torch.matmul(d_output, l2_w) 
        
#         # Gradients for the hidden layer (d_hidden from your numpy script)
#         d_hidden = error_hidden * self.sigmoid_derivative(activated_hidden)
        
#         # --- 4. Gradient Calculation for Weights ---
#         # (grad_weights_* from your numpy script)
        
#         # grad_l2_w = activated_hidden.T @ d_output
#         # (5, batch) @ (batch, 1) -> (5, 1) -> T -> (1, 5)
#         grad_l2_w = torch.matmul(activated_hidden.t(), d_output).t() 
#         grad_l2_b = torch.sum(d_output, dim=0) # (batch, 1) -> (1,)
        
#         # grad_l1_w = data.T @ d_hidden
#         # (2, batch) @ (batch, 5) -> (2, 5) -> T -> (5, 2)
#         grad_l1_w = torch.matmul(data.t(), d_hidden).t() 
#         grad_l1_b = torch.sum(d_hidden, dim=0) # (batch, 5) -> (5,)

#         # --- 5. Optimizer Step (Manual SGD) ---
#         # (The weight update from your numpy script)
        
#         # This is a pure calculation, so no_grad() is appropriate
#         with torch.no_grad():
#             new_l1_w = l1_w - self.lr * grad_l1_w
#             new_l1_b = l1_b - self.lr * grad_l1_b
#             new_l2_w = l2_w - self.lr * grad_l2_w
#             new_l2_b = l2_b - self.lr * grad_l2_b
            
#         # 6. Return the new weights
#         return new_l1_w, new_l1_b, new_l2_w, new_l2_b

# # --- Main Export Logic ---
# if __name__ == "__main__":
#     print("Exporting the *full functional* training step to ONNX...")

#     # 1. Instantiate the traceable module
#     export_model = FullTrainingStep()
#     export_model.eval() # Set to eval mode for tracing

#     # 2. Create dummy inputs to define the shapes
#     batch_size = 4 # From your XOR data
#     dummy_data = torch.randn(batch_size, INPUT_SIZE)
#     dummy_labels = torch.randn(batch_size, OUTPUT_SIZE)
    
#     # Create a temporary model just to get the initial weight shapes
#     temp_model = PyTorchNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    
#     # 3. Define all inputs for the function
#     all_inputs = (
#         dummy_data,
#         dummy_labels,
#         temp_model.layer1.weight,
#         temp_model.layer1.bias,
#         temp_model.layer2.weight,
#         temp_model.layer2.bias
#     )
    
#     # 4. Define the names for the input/output nodes
#     input_names = [
#         "data", "labels", 
#         "l1_w_in", "l1_b_in", 
#         "l2_w_in", "l2_b_in"
#     ]
    
#     output_names = [
#         "l1_w_out", "l1_b_out", 
#         "l2_w_out", "l2_b_out"
#     ]

#     # 5. Call the ONNX Exporter
#     torch.onnx.export(
#         export_model,          # The nn.Module instance (THIS IS THE FIX)
#         all_inputs,            # The dummy inputs
#         "xor_train_step.onnx", # The output file name
#         input_names=input_names,
#         output_names=output_names,
#         verbose=False 
#     )
    
#     print("\nSUCCESS!")
#     print("Created 'xor_train_step.onnx'.")
#     print("This file contains the *full* functional training step (fwd, loss, back, update).")
#     print("Your next step is to run the ZKTorch compiler on this file.")



import torch
import torch.onnx
import torch.nn as nn
from fl import PyTorchNN # Import your model

# --- Define Model Constants ---
INPUT_SIZE = 2
HIDDEN_SIZE = 5
OUTPUT_SIZE = 1
BATCH_SIZE = 4

print("Exporting PyTorch model in TRAINING mode...")

# 1. Instantiate your model
model = PyTorchNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.train() # Set model to training mode

# 2. Create dummy inputs
dummy_data = torch.randn(BATCH_SIZE, INPUT_SIZE)

# 3. Export with 'TrainingMode.TRAINING'
# This is the key. It saves the graph for training.
torch.onnx.export(
    model,
    (dummy_data,),
    "xor_inference_for_training.onnx", # This is just the forward pass
    input_names=["data"],
    output_names=["predictions"],
    training=torch.onnx.TrainingMode.TRAINING, # <-- THE MAGIC FLAG
    export_params=True,
    do_constant_folding=False,
    opset_version=12
)

print(f"SUCCESS: Created 'xor_inference_for_training.onnx'")
print("This file contains the forward pass, ready for the training generator.")