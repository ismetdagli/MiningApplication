# Mining Application and Overall Evaluation Experiments

Welcome to this machine learning and deep learning project. This repository contains various Python scripts that implement different machine learning algorithms and data handling routines. The code is built using PyTorch, with variations for CPU and GPU execution. Below is an overview of the files included in this project and their respective functions.

# Under scripts:

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

# Under experiments:

This project contains Python scripts focused on various aspects of mining and scheduling, with applications in edge and server environments. The scripts offer a range of functionalities, from mining simulations to different scheduling strategies, both locally and across clusters.

## File Descriptions

- **mining_scheduling_clusters_edgeandservers.py**
  - This script manages mining and scheduling across clusters, incorporating edge and server environments. It offers flexibility for both centralized and distributed configurations.

- **mining_scheduling_priority_asksameserver.py**
  - This script handles mining with scheduling that prioritizes tasks asking for the same server. It is useful for optimizing server load and minimizing data transfer times.

- **mining_scheduling_priority_dontaskedge.py**
  - This script is similar to the above, but with a scheduling priority that avoids asking edge servers. It is designed for scenarios where edge resources are limited or less reliable.

- **run_local.py**
  - This script is designed for running simulations or tasks locally. It is ideal for debugging or smaller-scale testing.

- **run_local_scheduler.py**
  - This script offers local scheduling capabilities, allowing for controlled experiments with different scheduling strategies.

- **simulation_mining.py**
  - This script simulates mining operations. It can be used to test different configurations and parameters in a controlled environment.

- **simulation_mining_strongscaling_v2.py**
  - A variant of the mining simulation script, designed for strong scaling. It tests how mining operations perform as resources are increased.

- **vr_scheduling_priority_dontaskedge.py**
  - This script manages VR scheduling with a priority to avoid edge servers. It is tailored for VR applications where latency and stability are crucial.

- **edge_server_count.py**
  - This script provides utilities for counting and managing edge servers. It can help assess the available resources and their utilization.

- **vr_weak_scaling.py**
  - This script is designed for VR applications with weak scaling, testing how VR tasks perform as additional resources are introduced.



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

