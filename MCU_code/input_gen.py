
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch

# Load the input from framework
input_tensor = np.loadtxt("../pc_code/input_sim.txt")
input_tensor = torch.from_numpy(input_tensor.reshape(3,128,128))
input_tensor = F.pad(input_tensor, (1, 0, 1, 0), value=0)
# Flatten the tensor into a 1D array
flattened_array = input_tensor.view(-1)

# Convert the flattened tensor to a list of float values
flattened_list = flattened_array.tolist()

# Perform the desired operations on each element
processed_values = [(x / 0.017818455 + 114.38545) for x in flattened_list]
processed_values = [min(max(round(value), 0), 255) for value in processed_values]

with open("input_new.txt", 'w') as input_file:
    for val in processed_values:
        input_file.write(str(val)+"\n")