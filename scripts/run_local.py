import socket
import time
import json

# Function to perform addition
def add_numbers(a, b):
    return a + b

# Total number of communication iterations
total_iterations = 100

# Create a socket (SOCK_STREAM means a TCP socket)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    # Bind the socket to an address and a port
    sock.bind(('138.67.22.142', 8888))  # Use the IP of the first device

    # Listen for incoming connections
    sock.listen()
    print("Waiting for a connection...")

    connection, client_address = sock.accept()
    with connection:
        print('Connected by', client_address)

        # Perform an addition
        result = add_numbers(5, 3)  # Example addition

        # Prepare a list of 100 elements, all being 2
        numbers_list = [2] * 50
        # Convert list to JSON string
        json_data = json.dumps(numbers_list)
        average_time=0

        for _ in range(total_iterations):

            # Start the timer
            start_time = time.time()

            # # Send numbers to the second device
            # connection.sendall(b'5 3')
            connection.sendall(json_data.encode())

            # Wait for the response from the second device
            data = connection.recv(1024)

            # Stop the timer
            end_time = time.time()

            # Calculate the time taken in milliseconds
            time_taken = (end_time - start_time) * 1000
            # print(f'Start time {_+1}:          {start_time} milliseconds')
            # print(f'Start time {_+1}:          {end_time} milliseconds')
            # print(f'Iteration {_+1}: Time taken {time_taken:.2f} milliseconds')
            average_time+=time_taken
            # received_result = int(data.decode())
            # if result == received_result:
            #     print("Results match.")
            # else:
            #     print("Results do not match.")
        print("Average time: ", average_time/total_iterations)
        print("Completed all iterations.")
