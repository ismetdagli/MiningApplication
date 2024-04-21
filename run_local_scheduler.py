import socket
import json
import time
from time import sleep


total_iterations = 1
cpu_available=True
gpu_available=True

def attempt_run_locally(task_function, min_time):
    max_time = 999
    device="none"
    if task_function == "rendering": # rendering on cpu: 100, gpu: 20
        if gpu_available:
            if min_time > 50:
                 max_time=50
                 device="gpu"
        if cpu_available:
            if min_time > 200:
                 max_time=100
                 device="cpu"
        return device,max_time
    


while True:
    # Start the timer
    total_start_time = time.time()

    device, shortest_time=attempt_run_locally("rendering",40.0)

    sleep(0.04)

    total_end_time = time.time()
    time_taken = (total_end_time - total_start_time) * 1000
    # print(f'Total time: {time_taken:.2f} milliseconds')
    # print("Operation completed.")