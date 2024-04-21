import socket
import time
import json

# Function to perform addition
def add_numbers(a, b):
    return a + b

def connect_to_server(address, retry_interval=0.1):
    """Attempt to connect to the server with rapid retries."""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(address)
            return sock
        except ConnectionRefusedError:
            # print("Connection failed. Retrying...")
            time.sleep(retry_interval)

def return_shortest(task_function):
    max_time = 999
    device="none"
    if task_function == "rendering": # rendering on cpu: 100, gpu: 20
        if gpu_available:
            if max_time > 20:
                 max_time=20
                 device="gpu"
        if cpu_available:
            if max_time > 100:
                 max_time=100
                 device="cpu"
        return device,max_time 
        

# Server's address and port
server_address = ('138.67.20.45', 5000)

# Connect to the server with rapid retries
sock = connect_to_server(server_address)
print("Connected to server.")

cpu_available=True
gpu_available=True

with sock:
    # for _ in range(100):  # Same number of iterations as the server
        # Receive data from the server
    data = sock.recv(2048)
    task_function, obj, constraints = json.loads(data.decode())
    print('Received data:', task_function, obj, constraints)
    
    device,shortest_time = return_shortest(task_function)

    print("shortest_time: ", shortest_time)
    if obj == "MIN_time":
        if constraints > shortest_time:
            response = "confirmed, "+str(shortest_time)
        else:
            response = "no device available, shortest: "+str(shortest_time)
    else:
        response = "invalid request"

    # Send the response back to the server
    sock.sendall(response.encode())
    print("Response sent:", response)

    # print("Completed all iterations.")
