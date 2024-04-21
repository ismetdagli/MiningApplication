import torch
import torch.nn as nn
import time

# Define the convolutional layer
# Here, we are assuming the input has 1 channel (like grayscale images), 
# we want to output 16 channels, and we are using a 3x3 kernel.
conv_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

# Explicitly move the layer to CPU
# This step is technically optional here since tensors and layers default to CPU,
# but it's good practice when working in mixed device environments.
conv_layer = conv_layer.to('cpu')

# Define an input tensor
# Let's say we have a batch of 8 images, each 1 channel (grayscale), 28x28 pixels.
# The tensor shape is (batch_size, channels, height, width).

input_tensor = torch.randn(1, 1, 2048, 2048)

# Explicitly move the input tensor to CPU (if it's not already)
input_tensor = input_tensor.to('cpu')

for i in range(5):
    output_tensor = conv_layer(input_tensor)

# Measure the start time
start_time = time.time()

# Apply the convolutional layer to the input tensor
for i in range(1000000):
    output_tensor = conv_layer(input_tensor)


# Measure the end time
end_time = time.time()
# Calculate the duration in milliseconds
duration_ms = (end_time - start_time) * 1000

# Output tensor shape will be (batch_size, out_channels, output_height, output_width),
# which depends on the stride, padding, and kernel size.
print("Output tensor shape:", output_tensor.shape)

# Print the duration of the operation
print(f"Operation took {duration_ms:.2f} milliseconds.")