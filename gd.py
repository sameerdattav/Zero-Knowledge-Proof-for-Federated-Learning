import numpy as np
import torch

def generate_toy_data(num_points=4):
    # Simple dataset; you can change numbers as needed
    full_x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    full_y = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=float)  # y = 2*x
    return full_x[:num_points], full_y[:num_points]

class ToyLinearModel:
    def __init__(self):
        self.w = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
        self.b = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.w * x + self.b

    def gd_step(self, x, y, lr=0.01):
        x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        y_t = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

        preds = self.forward(x_t)
        loss = torch.mean((preds - y_t) ** 2)
        loss.backward()

        # read grads before update
        grad_w = float(self.w.grad.item())
        grad_b = float(self.b.grad.item())

        with torch.no_grad():
            self.w -= lr * self.w.grad
            self.b -= lr * self.b.grad

        self.w.grad.zero_()
        self.b.grad.zero_()

        return grad_w
