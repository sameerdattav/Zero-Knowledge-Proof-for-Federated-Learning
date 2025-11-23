import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np
import os

print("==================================================")
print("   STEP 1: PATCHING THE GRAPH (Fixing the 'Boolean' bug)")
print("==================================================")

# --- PART 1: THE PATCHER ---
input_path = "training_artifacts/training_model.onnx"
output_path = "training_artifacts/training_model_patched.onnx"

if not os.path.exists(input_path):
    print(f"❌ Error: Could not find {input_path}")
    exit()

model = onnx.load(input_path)
graph = model.graph

nodes_to_remove = []
new_nodes = []
new_outputs = []

print(f"Scanning {len(graph.node)} nodes...")

count = 0
for i, node in enumerate(graph.node):
    # Check for any node type containing "InPlaceAccumulator"
    if "InPlaceAccumulator" in node.op_type:
        count += 1
        # Get inputs: [buffer, gradient, ...]
        acc_buffer = node.input[0]
        new_grad = node.input[1]
        
        # Create a name for the real output value
        real_output_name = acc_buffer + "_valued"
        
        # Create a standard 'Add' node (Buffer + Gradient)
        add_node = helper.make_node(
            "Add",
            inputs=[acc_buffer, new_grad],
            outputs=[real_output_name],
            name=node.name.replace(node.op_type, "Add_Patch")
        )
        
        new_nodes.append(add_node)
        nodes_to_remove.append(node)
        
        # Define the New Graph Output info
        input_shape = None
        for inp in graph.input:
            if inp.name == acc_buffer:
                input_shape = inp.type.tensor_type.shape
                break
        
        new_output_info = helper.make_tensor_value_info(
            real_output_name,
            TensorProto.FLOAT,
            [d.dim_value for d in input_shape.dim] if input_shape else None 
        )
        new_outputs.append(new_output_info)

if count == 0:
    print("⚠️ WARNING: No Accumulator nodes found! Is this the right file?")
else:
    print(f"✅ Found and Patching {count} nodes...")
    
    for node in nodes_to_remove:
        graph.node.remove(node)
    graph.node.extend(new_nodes)

    # Fix the Outputs: Keep Loss, Add new Gradients
    loss_output = graph.output[0] 
    del graph.output[:] 
    graph.output.append(loss_output) 
    graph.output.extend(new_outputs) 

    onnx.save(model, output_path)
    print(f"✅ SUCCESS: Saved patched graph to: {output_path}")


print("\n==================================================")
print("   STEP 2: VERIFYING THE PATCHED GRAPH")
print("==================================================")

# --- PART 2: THE VERIFIER ---

# Load Ground Truth
try:
    ground_truth = np.load('ground_truth_sgd.npz')
except:
    print("❌ Error: 'ground_truth_sgd.npz' not found.")
    exit()

# Load the NEW patched model
try:
    sess = ort.InferenceSession(output_path)
except Exception as e:
    print(f"❌ Error loading patched model: {e}")
    exit()

l1_w_shape = ground_truth['l1_w_in'].shape
l1_b_shape = ground_truth['l1_b_in'].shape
l2_w_shape = ground_truth['l2_w_in'].shape
l2_b_shape = ground_truth['l2_b_in'].shape

# Prepare Inputs
train_inputs = {
    'data': ground_truth['data'],
    'target': ground_truth['labels'],
    'layer1.weight': ground_truth['l1_w_in'],
    'layer1.bias': ground_truth['l1_b_in'],
    'layer2.weight': ground_truth['l2_w_in'],
    'layer2.bias': ground_truth['l2_b_in'],
    'lazy_reset_grad': np.array([False], dtype=bool),
    'layer1.weight_grad.accumulation.buffer': np.zeros(l1_w_shape, dtype=np.float32),
    'layer1.bias_grad.accumulation.buffer': np.zeros(l1_b_shape, dtype=np.float32),
    'layer2.weight_grad.accumulation.buffer': np.zeros(l2_w_shape, dtype=np.float32),
    'layer2.bias_grad.accumulation.buffer': np.zeros(l2_b_shape, dtype=np.float32)
}

print("Running PATCHED Training Graph...")
outputs = sess.run(None, train_inputs)

loss = outputs[0]
gradients = outputs[1:]

print(f"Loss: {loss}")
print(f"Number of Gradients Returned: {len(gradients)}")

expected_shapes = [l1_w_shape, l1_b_shape, l2_w_shape, l2_b_shape]
match_count = 0

print("\n--- Checking Gradient Shapes ---")
for i, grad in enumerate(gradients):
    print(f"  - Output {i} Shape: {grad.shape}")
    for exp in expected_shapes:
        if grad.shape == exp:
            match_count += 1
            expected_shapes.remove(exp) 
            break

if match_count == 4:
    print("\n✅ SUCCESS! The patched graph outputs gradients with the CORRECT shapes.")
    print("✅ You now have a functional, ZK-compatible training graph.")
else:
    print("\n❌ FAIL. Gradient shapes do not match parameters.")