import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18
import torch
import torch.nn as nn
import json
import numpy as np
from replace_relu import replace_relu_attr
from intermerdiate_output_hooks import IntermediateOutputsHook

def trace_and_save_data(hook):
    mapping = {}
    layer_id = 0
    for layer in zip(hook.inputs, hook.outputs, hook.modules):
        layer_id += 1

        if isinstance(layer[2], nn.Conv2d):
            kernel_size = layer[2].kernel_size
            padding = layer[2].padding
            groups = layer[2].groups
            stride = layer[2].stride
            if layer[2].bias != None:
                bias = layer[2].bias.detach().numpy().tolist()
            else:
                bias = []
            c, h, w = layer[1][0].shape
            b, c1, h1, w1 = layer[0][0].shape
            weights = layer[2].weight.detach().numpy().tolist()
            input_per_group = int(c1 / groups)
            output_per_group = int(c / groups)
            o_i_mapping = {
                "o_pg": output_per_group,
                'i_pg': input_per_group,
                "s": stride,
                "k": kernel_size,
                "i": (c1, h1, w1),
                "o": (c, h, w),
            }
            mapping[f"{layer_id}"] = {"Convolution": {"w": weights, "info": o_i_mapping, "bias": bias}}

        if isinstance(layer[2], torch.nn.BatchNorm2d):
            weights = layer[2].weight.detach().tolist()
            bias = layer[2].bias.detach().tolist()
            r_m = layer[2].running_mean.detach().tolist()
            r_v = layer[2].running_var.detach().tolist()
            input_shape = layer[0][0].shape
            mapping[f"{layer_id}"] = {
            "BatchNorm2d": {"w": weights, "bias": bias, "r_m": r_m, "r_v": r_v, "input_shape": input_shape}}

        if isinstance(layer[2], torch.nn.ReLU6):
            input_shape = layer[0][0].shape
            mapping[f"{layer_id}"] = {"ReLU6": {"input_shape": input_shape}}

        if layer_id == 3:
            np.savetxt("../test_references/test_cbr_resnet18.txt", layer[1][0].flatten().detach().numpy(), fmt='%.10f', delimiter=',')
            break

    return mapping

# Load the pretrained ResNet model
model = resnet18(weights="DEFAULT")
model.eval()

#Replace the AdaptiveAvgPool layer with AvgPool
#model.avgpool = torch.nn.AvgPool2d((7,7),512)
#switch out ReLu with ReLu6
replace_relu_attr(model)

# Instantiate the hook
hook = IntermediateOutputsHook()
hook.register(model)

# Custom input tensor - dimensions same as `test_convolution()` in `pc_code/Algorithms/src/main.rs`
input_data = torch.zeros((1, 3, 44, 44))

# Populate the tensor with the desired values (Rust code does the same thing)
for c in range(3):
    for i in range(44):
        input_data[0, c, i, :] = torch.tensor([float(i) for _ in range(44)], dtype=torch.float64)

output = model(input_data)

# get weights, save
mapping = trace_and_save_data(hook)
with open('../json_files/test_resnet18_cbr.json', 'w') as file:
    json.dump(mapping, file)
print("-----")
# Remove the hooks after you're done
hook.remove_hooks()