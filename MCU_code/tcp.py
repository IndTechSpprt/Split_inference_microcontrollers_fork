import socket
import select
import time
import numpy as np
from PIL import Image
from torchvision import transforms
import struct
import torch.nn.functional as F
import json

# input_image = Image.open("./img.png")
# input_image = input_image.convert("RGB")
#
# # Define the preprocessing transformation
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# # Apply the preprocessing transformation
# input_tensor = preprocess(input_image)
# input_tensor = F.pad(input_tensor, (0, 1, 0, 1), value=0)
# # Flatten the tensor into a 1D array
# flattened_array = input_tensor.view(-1)
#
# # Convert the flattened tensor to a list of float values
# flattened_list = flattened_array.tolist()
#
# # Perform the desired operations on each element
# processed_values = [(x / 0.017818455 + 114.38545) for x in flattened_list]
# processed_values = [min(max(round(value), 0), 255) for value in processed_values]
with open('input.txt', 'r') as file:
    # Read lines from the file and strip newline characters
    lines = file.readlines()
    # Convert each line to an integer and collect them into a list
    processed_values = [int(line.strip()) for line in lines]
print("!")
processed_values = bytearray(processed_values)

#Testbed
with open("testbed.json", 'r') as file:
    testbed = json.load(file)

# PC's IP address and port
message_size = 1400
reserved_bytes = 6
num_mcu = len(testbed)
pc_ip = "169.254.71.125"  # Replace with PC's IP address
pc_port = 8080  # Replace with PC's port number
ip_first_part = "169.254.71."
# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
server_socket.bind((pc_ip, pc_port))

# Listen for incoming connections
server_socket.listen(num_mcu)
print("Waiting for connection...")
sockets = [None] * num_mcu
addresses = [None] * num_mcu
which = -1
for count in range(num_mcu):
    # Accept a connection
    client_socket, client_address = server_socket.accept()
    client_socket.setblocking(0)  # Set socket to non-blocking mode
    print(str(client_address[0]))
    for mcu in testbed:
        if str(client_address[0]) == ip_first_part+mcu["ip_end"]:
            which = int(mcu["mcu_id"])
    sockets[which] = client_socket
    addresses[which] = client_address
    print(f"Connected to Arduino {which} at:", client_address)
for s in sockets:
    s.sendall(b'\x41')  # signal of connection success
print("connection established!")
working = False
pending = []
cur = 0
layer_levels = [0] * num_mcu
last_send_time = time.time()
ep = 0
data_to_send = bytearray(message_size)
from_which = -1
input_flag = False


def wait_for_ack(sock, message_size):
    while True:
        print("waiting for ack")
        socketslist = [sock]
        read, _, _ = select.select(socketslist, [], [])
        if len(read) > 0:
            for s in read:
                message = bytearray(s.recv(message_size))
                if message[1] == 197:
                    print("ack received!")
                    return
                elif message[1] == 199:
                    print(f"received request from {message[0]}")
                    pending.append([layer_levels[message[0]], message[0]])
                    pending.sort()
                    layer_levels[message[0]] += 1
                    print(pending)


def send_ack(sock, data):
    print("sending ack")
    data[1] = 197
    data = data[:3]
    sock.sendall(data)

pooled_means = []
to_adaptive_pooling = []
#
for i in range(0, len(processed_values), message_size - reserved_bytes):
    message = [0] * message_size
    message[0] = 10  # from coordinator
    integer_value = min(message_size - reserved_bytes, len(processed_values) - i)
    # Pack the integer into 4 bytes (assuming it's a 32-bit integer)
    packed_bytes = struct.pack('<i', integer_value)
    # Assign the packed bytes to indices 2 to 6 of the byte array
    message[2:reserved_bytes] = packed_bytes
    message[reserved_bytes:reserved_bytes + integer_value] = processed_values[i: i + integer_value]
    # for j in range(len(sockets)):
    message = bytearray(message)
    print(message)
    print("\n")
    for i in range(len(sockets)):
        message[1] = i
        sockets[i].sendall(message)
        wait_for_ack(sockets[i], message_size)
print("start coordinating...")
try:
    while True:
        readable, _, _ = select.select(sockets, [], [])  # Select sockets ready to read
        for sock in readable:
            for count in range(num_mcu):
                if sock == sockets[count]:
                    try:
                        # time.sleep(0.01)
                        received_data = sock.recv(message_size)
                    except:
                        continue
                        # received_data = sock.recv(message_size)
                    received_data = bytearray(received_data)
                    if received_data[1] != 199:
                        send_ack(sock, received_data.copy())
                    byte_array = bytes(received_data)
                    print(f"len of data is:{len(byte_array)}")
                    from_which = received_data[0]
                    if received_data[1] == 199:
                        print(f"received request from{from_which}")
                        pending.append([layer_levels[from_which], from_which])
                        pending.sort()
                        print(pending)
                        layer_levels[from_which] += 1
                        if not working:
                            next = pending[0][1]
                            if next == ep:
                                working = True
                                ep += 1
                                if ep == num_mcu:
                                    ep = 0
                                received_data = bytearray(received_data)
                                received_data[1] = 200
                                print(f"sending permission to {next}")
                                sockets[next].sendall(received_data)
                                wait_for_ack(sockets[next], message_size)
                    elif received_data[1] == 198:
                        print(pending)
                        print(ep)
                        pending.pop(0)
                        working = False
                        received_data[1] = 200
                        if len(pending) != 0 and ep == pending[0][1]:
                            ep += 1
                            if ep == num_mcu:
                                ep = 0
                            next = pending[0]
                            print(pending)
                            print(f"next: {next}")
                            working = True
                            print(f"sending permission to {next}")
                            sockets[next[1]].sendall(received_data)
                            wait_for_ack(sockets[next[1]], message_size)
                        if len(to_adaptive_pooling) == 1280 * 7 * 7:
                            for i in range(0, len(to_adaptive_pooling), 7 * 7):
                                chunk = to_adaptive_pooling[i:i + 7 * 7]
                                average = int(round(sum(chunk) / len(chunk)))
                                pooled_means.append(average)
                            for i in range(len(sockets)):
                                message = [0] * 1286
                                message[0] = 10
                                message[1] = i
                                packed_bytes = struct.pack('<i', 1280)
                                message[2:6] = packed_bytes
                                message[6: 1286] = pooled_means[:]
                                message = bytearray(message)
                                sockets[i].sendall(message)
                                wait_for_ack(sockets[i],message_size)

                    elif received_data[1] == 196:
                        len_int = struct.unpack('<I', received_data[2:6])[0]
                        print(f"received pooling data from MCU{received_data[0]}, len {len_int}")
                        for i in range(6, 6 + len_int):
                            to_adaptive_pooling.append(received_data[i])
                        print(len(to_adaptive_pooling))

                    else:
                        data_to_send = received_data
                        mcus = data_to_send[1]
                        to_which = []
                        print(mcus)
                        for i in range(num_mcu):
                            if (mcus >> i & 0b1) == 1:
                                to_which.append(i)
                        print(f"Received from Arduino{from_which} to {to_which}:")
                        for w in to_which:
                            data_to_send[1] = w
                            sockets[w].sendall(data_to_send)
                        for w in to_which:
                            wait_for_ack(sockets[w], message_size)
except KeyboardInterrupt:
    print("Closing connection")
