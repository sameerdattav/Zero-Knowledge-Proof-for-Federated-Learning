from onnxruntime.training import artifacts
import onnx

print("Loading forward-pass ONNX model...")
# Load the model you just exported
onnx_model = onnx.load("xor_inference_for_training.onnx")

# Define what parameters need gradients
# NEW LINE (CORRECT):
requires_grad = [param.name for param in onnx_model.graph.initializer]
# Define a loss function
# We can tell it to add a standard loss to the graph
loss = artifacts.LossType.MSELoss

# Tell it to create an optimizer graph
optimizer = artifacts.OptimType.AdamW

print("Starting artifact generation...")
# This is the magic function that does all the work!
# It auto-generates the backward pass, loss, and optimizer
artifacts.generate_artifacts(
    onnx_model,
    requires_grad=requires_grad,
    loss=loss,
    optimizer=optimizer,
    artifact_directory="training_artifacts" # Output folder
)

print("\nSUCCESS!")
print("Created the 'training_artifacts' directory with all training graphs:")
print("- train_model.onnx (Forward + Loss + Backward graphs)")
print("- optimizer_model.onnx (The optimizer graph)")
print("- checkpoint (The model's initial weights)")