import torch
import torch.nn as nn
import time

# Check if CUDA (GPU support) is available, otherwise revert to CPU
device = torch.device("cuda")
print(f"Running on device: {device}")

# Define the convolutional layer
conv_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

# Move the convolutional layer to the specified device (GPU if available)
conv_layer = conv_layer.to(device)

# Define an input tensor
input_tensor = torch.randn(8, 1, 2048, 2048)

# Move the input tensor to the specified device (GPU if available)
input_tensor = input_tensor.to(device)

for i in range(5):
    output_tensor = conv_layer(input_tensor)

# Measure the start time
start_time = time.time()

# Apply the convolutional layer to the input tensor
for i in range(100):
    output_tensor = conv_layer(input_tensor)

# Wait for all GPU kernels to complete before measuring the end time
torch.cuda.synchronize(device)

# Measure the end time
end_time = time.time()

# Calculate the duration in milliseconds
duration_ms = (end_time - start_time) * 1000

# Output tensor shape
print("Output tensor shape:", output_tensor.shape)

# Print the duration of the operation
print(f"Operation took {duration_ms:.2f} milliseconds.")
