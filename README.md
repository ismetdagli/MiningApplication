# Mining Application

Welcome to this machine learning and deep learning project. This repository contains various Python scripts that implement different machine learning algorithms and data handling routines. The code is built using PyTorch, with variations for CPU and GPU execution. Below is an overview of the files included in this project and their respective functions.

## File Descriptions

- **main.py**
  - This is the primary entry point for the project. It initializes the project and calls other scripts to execute specific tasks.

- **loader.py**
  - Responsible for loading and preprocessing the datasets used throughout the project.

- **convolution_torch_cpu.py**
  - Implements convolutional neural networks (CNNs) with PyTorch for CPU execution. Use this script for running deep learning models without GPU acceleration.

- **convolution_torch_gpu.py**
  - Similar to `convolution_torch_cpu.py`, but optimized for GPU execution. It requires CUDA and an appropriate GPU to run efficiently.

- **knn_pytorch_v2.py**
  - Contains an implementation of the K-nearest neighbors (KNN) algorithm using PyTorch for CPU.

- **knn_pytorch_v2gpu.py**
  - Like `knn_pytorch_v2.py`, but designed for GPU execution. Use this for accelerated KNN computations.

- **mlp_torch.py**
  - Implements a multi-layer perceptron (MLP) in PyTorch. It can run on both CPU and GPU.

- **thundersvm_cpu.py**
  - Implements support vector machines (SVMs) for CPU, using ThunderSVM. This script is useful for classical machine learning tasks without GPU acceleration.

- **thundersvm_gpu.py**
  - Implements SVMs using ThunderSVM with GPU acceleration. This script is optimized for running SVMs on GPU.

- **load_and_plot_results.py**
  - This script loads results from the other scripts and creates various plots to visualize the output. Useful for analyzing and understanding the performance of different algorithms.

## Running the Scripts

To run any of the scripts, ensure you have Python installed along with the required libraries, such as PyTorch and ThunderSVM. Additionally, if you're running GPU-based scripts, ensure your environment supports CUDA.

1. **Setting Up Environment**
   - Install Python 3.6.
   - Install required libraries: `pip install torch thundersvm`.

2. **Running a Script**
   - Navigate to the project directory.
   - Execute the desired script: `python script_name.py`.

## Additional Notes

- Ensure you have the correct hardware setup for GPU-based scripts.
- Before running GPU-based scripts, confirm your system supports CUDA.
- If you're processing large datasets, ensure you have sufficient memory and computational resources.

