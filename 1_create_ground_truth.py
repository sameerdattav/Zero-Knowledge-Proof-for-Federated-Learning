import torch
import numpy as np
from fl import PyTorchNN  # Import your PyTorch model

# --- Define Model Constants ---
INPUT_SIZE = 2
HIDDEN_SIZE = 5
OUTPUT_SIZE = 1
LEARNING_RATE = 0.1
BATCH_SIZE = 4

print("Running one PyTorch step to create 'ground_truth.npz'...")

# 1. Instantiate the PyTorch model
model = PyTorchNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.train() # Set to training mode

# 2. Create deterministic dummy data
# We use torch.manual_seed to make sure the data is the same every time
torch.manual_seed(42)
data = torch.randn(BATCH_SIZE, INPUT_SIZE)
labels = torch.randn(BATCH_SIZE, OUTPUT_SIZE)

# Get the initial weights
l1_w_in = model.layer1.weight.data.clone()
l1_b_in = model.layer1.bias.data.clone()
l2_w_in = model.layer2.weight.data.clone()
l2_b_in = model.layer2.bias.data.clone()

# 3. Run the *real* PyTorch training step
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

optimizer.zero_grad()
y_hat = model(data)
loss = criterion(y_hat, labels)
loss.backward()
optimizer.step()

# 4. Get the *new* weights (the "correct answer")
l1_w_out = model.layer1.weight.data.clone()
l1_b_out = model.layer1.bias.data.clone()
l2_w_out = model.layer2.weight.data.clone()
l2_b_out = model.layer2.bias.data.clone()

# 5. Save all inputs and outputs to a file
# We convert to numpy for universal compatibility
np.savez(
    'ground_truth.npz',
    # --- Inputs ---
    data=data.numpy(),
    labels=labels.numpy(),
    l1_w_in=l1_w_in.numpy(),
    l1_b_in=l1_b_in.numpy(),
    l2_w_in=l2_w_in.numpy(),
    l2_b_in=l2_b_in.numpy(),
    # --- Correct Outputs ---
    l1_w_out_gt=l1_w_out.numpy(),
    l1_b_out_gt=l1_b_out.numpy(),
    l2_w_out_gt=l2_w_out.numpy(),
    l2_b_out_gt=l2_b_out.numpy()
)

print("SUCCESS: 'ground_truth.npz' created.")