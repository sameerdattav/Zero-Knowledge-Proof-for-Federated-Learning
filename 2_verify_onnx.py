import onnxruntime as ort
import numpy as np

# We know the LR from our PyTorch script
LEARNING_RATE = 0.1

print("Loading 'ground_truth.npz' and ALL training artifacts...")

# 1. Load the ground truth data *FIRST*
ground_truth = np.load('ground_truth.npz')

# 2. Load BOTH ONNX models
try:
    sess_train = ort.InferenceSession('training_artifacts/training_model.onnx')
    sess_optim = ort.InferenceSession('training_artifacts/optimizer_model.onnx')
except Exception as e:
    print(f"Error loading ONNX models: {e}")
    print("Please ensure 'training_artifacts' folder is correct.")
    exit()

# 3. Get shapes from the ground truth data
l1_w_shape = ground_truth['l1_w_in'].shape
l1_b_shape = ground_truth['l1_b_in'].shape
l2_w_shape = ground_truth['l2_w_in'].shape
l2_b_shape = ground_truth['l2_b_in'].shape

# --- Create Zero-Buffers for AdamW State (for Step 0) ---
zero_buf_l1_w = np.zeros(l1_w_shape, dtype=np.float32)
zero_buf_l1_b = np.zeros(l1_b_shape, dtype=np.float32)
zero_buf_l2_w = np.zeros(l2_w_shape, dtype=np.float32)
zero_buf_l2_b = np.zeros(l2_b_shape, dtype=np.float32)

# Create *separate* zero-buffers for the second moment
zero_buf_v_l1_w = np.zeros(l1_w_shape, dtype=np.float32)
zero_buf_v_l1_b = np.zeros(l1_b_shape, dtype=np.float32)
zero_buf_v_l2_w = np.zeros(l2_w_shape, dtype=np.float32)
zero_buf_v_l2_b = np.zeros(l2_b_shape, dtype=np.float32)


# 4. Prepare inputs for the *TRAINING* model (fwd + loss + back)
train_inputs = {
    'data': ground_truth['data'],
    'target': ground_truth['labels'],
    'layer1.weight': ground_truth['l1_w_in'],
    'layer1.bias': ground_truth['l1_b_in'],
    'layer2.weight': ground_truth['l2_w_in'],
    'layer2.bias': ground_truth['l2_b_in'],
    'lazy_reset_grad': np.array([False], dtype=bool),
    'layer1.weight_grad.accumulation.buffer': zero_buf_l1_w,
    'layer1.bias_grad.accumulation.buffer': zero_buf_l1_b,
    'layer2.weight_grad.accumulation.buffer': zero_buf_l2_w,
    'layer2.bias_grad.accumulation.buffer': zero_buf_l2_b
}

print("Running the TRAINING graph (fwd + loss + back)...")
# 5. Run the TRAINING graph. This will output LOSS and GRADIENTS.
train_outputs = sess_train.run(None, train_inputs)

# 6. Extract the gradients
try:
    loss_onnx = train_outputs[0]
    grad_1 = train_outputs[1]
    grad_2 = train_outputs[2]
    grad_3 = train_outputs[3]
    grad_4 = train_outputs[4]
    print(f"ONNX Loss: {loss_onnx}")
    
    # --- THIS IS THE NEW DEBUG CODE ---
    print("\n--- DEBUG: Inspecting Gradient Shapes ---")
    print(f"Ground Truth Weight Shapes (Params):")
    print(f"  l1_w_in: {ground_truth['l1_w_in'].shape}")
    print(f"  l1_b_in: {ground_truth['l1_b_in'].shape}")
    print(f"  l2_w_in: {ground_truth['l2_w_in'].shape}")
    print(f"  l2_b_in: {ground_truth['l2_b_in'].shape}")
    
    print(f"\nTraining Graph Output Shapes (Gradients):")
    print(f"  train_outputs[1] (grad_1): {grad_1.shape}")
    print(f"  train_outputs[2] (grad_2): {grad_2.shape}")
    print(f"  train_outputs[3] (grad_3): {grad_3.shape}")
    print(f"  train_outputs[4] (grad_4): {grad_4.shape}")
    
    # ---
    # Based on the error, the order is probably wrong.
    # The error says param[0] (5,2) != grad[0] (shape of grad_1)
    # This implies grad_1 is NOT the l1_w_grad.
    # The order of params is: l1_w (5,2), l1_b (5,), l2_w (1,5), l2_b (1,)
    # Let's find the gradients that match this order
    # ---
    
    all_grads = [grad_1, grad_2, grad_3, grad_4]
    
    # We must re-order the `all_grads` list to match the param order
    # This is a robust way to fix the "mismatch" error
    
    grad_l1_w = next(g for g in all_grads if g.shape == l1_w_shape)
    grad_l1_b = next(g for g in all_grads if g.shape == l1_b_shape)
    grad_l2_w = next(g for g in all_grads if g.shape == l2_w_shape)
    grad_l2_b = next(g for g in all_grads if g.shape == l2_b_shape)
    
    print("\nSuccessfully re-ordered gradients based on shape.")

except Exception as e:
    print(f"\n❌ ERROR during gradient extraction: {e}")
    exit()

# 7. Prepare inputs for the *OPTIMIZER* model
#    This uses the *re-ordered* gradients
optim_inputs = {
    'learning_rate': np.array([LEARNING_RATE], dtype=np.float32),
    'step': np.array([0], dtype=np.int64),
    
    'params': [
        ground_truth['l1_w_in'].astype(np.float32),
        ground_truth['l1_b_in'].astype(np.float32),
        ground_truth['l2_w_in'].astype(np.float32),
        ground_truth['l2_b_in'].astype(np.float32)
    ],
    
    'gradients': [
        grad_l1_w.astype(np.float32), # Use re-ordered grad
        grad_l1_b.astype(np.float32), # Use re-ordered grad
        grad_l2_w.astype(np.float32), # Use re-ordered grad
        grad_l2_b.astype(np.float32)  # Use re-ordered grad
    ],
    
    'first_order_moments': [
        zero_buf_l1_w,
        zero_buf_l1_b,
        zero_buf_l2_w,
        zero_buf_l2_b
    ],
    'second_order_moments': [
        zero_buf_v_l1_w,
        zero_buf_v_l1_b,
        zero_buf_v_l2_w,
        zero_buf_v_l2_b
    ]
}

print("Running the OPTIMIZER graph...")
# 8. Run the OPTIMIZER graph.
try:
    optim_outputs = sess_optim.run(None, optim_inputs)
except Exception as e:
    print(f"\n❌ ERROR running the OPTIMIZER graph: {e}")
    print("This is likely a new input name mismatch. Please paste this error.")
    exit()

# 9. Get the final new weights
l1_w_out_onnx = optim_outputs[0]
l1_b_out_onnx = optim_outputs[1]
l2_w_out_onnx = optim_outputs[2]
l2_b_out_onnx = optim_outputs[3]

# 10. Compare!
print("Comparing ONNX outputs to PyTorch ground truth...")
try:
    np.testing.assert_allclose(ground_truth['l1_w_out_gt'], l1_w_out_onnx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ground_truth['l1_b_out_gt'], l1_b_out_onnx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ground_truth['l2_w_out_gt'], l2_w_out_onnx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ground_truth['l2_b_out_gt'], l2_b_out_onnx, rtol=1e-5, atol=1e-5)
    
    print("\n" + "="*30)
    print("    ✅ SUCCESS! ✅")
    print("="*30)
    print("The *full* ONNX pipeline (train + optim) output *exactly* matches the PyTorch ground truth.")
    print("Your artifacts are 100% correct and ready for ZKTorch.")

except AssertionError as e:
    print("\n" + "="*30)
    print("    ❌ VALIDATION FAILED ❌")
    print("="*30)
    print("The ONNX graph output does NOT match the PyTorch ground truth.")
    print("Error details:", e)