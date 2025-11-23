
import torch
import numpy as np
from fl import PyTorchNN

INPUT_SIZE = 2
HIDDEN_SIZE = 5
OUTPUT_SIZE = 1
LEARNING_RATE = 0.1
BATCH_SIZE = 4

print("Running one PyTorch SGD step...")

model = PyTorchNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.train()

torch.manual_seed(42)
data = torch.randn(BATCH_SIZE, INPUT_SIZE)
labels = torch.randn(BATCH_SIZE, OUTPUT_SIZE)

l1_w_in = model.layer1.weight.data.clone()
l1_b_in = model.layer1.bias.data.clone()
l2_w_in = model.layer2.weight.data.clone()
l2_b_in = model.layer2.bias.data.clone()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

optimizer.zero_grad()
y_hat = model(data)
criterion = torch.nn.MSELoss()
loss = criterion(y_hat, labels) # Calculate loss
loss.backward()
optimizer.step()

# --- NEW: Print the Ground Truth Loss ---
print(f"PyTorch Loss: {loss.item()}")

l1_w_out = model.layer1.weight.data.clone()
l1_b_out = model.layer1.bias.data.clone()
l2_w_out = model.layer2.weight.data.clone()
l2_b_out = model.layer2.bias.data.clone()

# --- NEW: Save 'loss_gt' to the file ---
np.savez(
    'ground_truth_sgd.npz',
    data=data.numpy(),
    labels=labels.numpy(),
    l1_w_in=l1_w_in.numpy(),
    l1_b_in=l1_b_in.numpy(),
    l2_w_in=l2_w_in.numpy(),
    l2_b_in=l2_b_in.numpy(),
    l1_w_out_gt=l1_w_out.numpy(),
    l1_b_out_gt=l1_b_out.numpy(),
    l2_w_out_gt=l2_w_out.numpy(),
    l2_b_out_gt=l2_b_out.numpy(),
    loss_gt=loss.detach().numpy() # <--- Saved here!
)
print("SUCCESS: 'ground_truth_sgd.npz' created.")