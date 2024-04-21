import socket
import json
import time

def create_server_socket(port):
    # Create a socket (SOCK_STREAM means a TCP socket)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('138.67.212.13', port))  # Use the IP of the first device
    server_socket.listen(2)  # Set up the server to accept 2 connections
    return server_socket

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

def send_data_to_client(connection):
    # Assign variables
    task_function = "rendering"
    obj = "MIN_time"
    constraints = 40.0

    # Prepare the data tuple
    data_tuple = (task_function, obj, constraints)
    # Convert tuple to JSON string
    json_data = json.dumps(data_tuple)

    # Start the timer
    start_time = time.time()

    # Send the tuple to the client device
    connection.sendall(json_data.encode())

    # Wait for the response from the client device
    data = connection.recv(2048)

    # Stop the timer
    end_time = time.time()

    # Calculate the time taken in milliseconds
    time_taken = (end_time - start_time) * 1000
    print(f'Time taken: {time_taken:.2f} milliseconds')

    received_result = data.decode()
    print(f'Received response from the client device: {received_result}')

server_socket = create_server_socket(8888)
server_socket2 = create_server_socket(5000)


# Total number of communication iterations, device 
total_iterations = 1
cpu_available=True
gpu_available=True
device, shortest_time=attempt_run_locally("rendering",40.0)
print ("device: ", device, " shortest_time: ",shortest_time)

if device == "none":
    print("Waiting for connections...")

    
# Start the timer
total_start_time = time.time()

    # Handle connection with the second device
try:
    connection, client_address = server_socket.accept()
    print('Connected by', client_address)
    send_data_to_client(connection)
    connection.close()
except Exception as e:
    print("Error or no connection from the second device.")

total_end_time = time.time()
time_taken = (total_end_time - total_start_time) * 1000
print(f'Total time: {time_taken:.2f} milliseconds')
# Handle connection with the third device
try:
    connection, client_address = server_socket2.accept()
    print('Connected by', client_address)
    send_data_to_client(connection)
    connection.close()
except Exception as e:
    print("Error or no connection from the third device.")


# Stop the timer
total_end_time = time.time()
time_taken = (total_end_time - total_start_time) * 1000
print(f'Total time: {time_taken:.2f} milliseconds')
print("Operation completed.")
