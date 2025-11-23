from onnxruntime.training import artifacts
import onnx

print("Loading forward-pass ONNX model...")
onnx_model = onnx.load("xor_inference_for_training.onnx")

requires_grad = [param.name for param in onnx_model.graph.initializer]
loss = artifacts.LossType.MSELoss

# --- THIS IS THE CHANGE ---
# We are DELETING the 'optimizer = ...' line.
# The library will default to SGD.
# --------------------------

print("Starting artifact generation (for SGD)...")
# The 'optimizer' argument is now gone.
artifacts.generate_artifacts(
    onnx_model,
    requires_grad=requires_grad,
    loss=loss,
    artifact_directory="training_artifacts" # <-- New folder
)

print("\nSUCCESS!")
print("Created 'training_artifacts' with (hopefully) SGD graphs.")

