import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import alexnet
import torch
import torch.nn as nn
import json
import numpy as np
from torchvision import transforms
from replace_relu import replace_relu_mod, replace_relu_attr

class IntermediateOutputsHook:
    def __init__(self):
        self.outputs = []
        self.handles = []
        self.inputs = []
        self.modules = []

    def register(self, model):
        # Register a forward hook for each submodule
        for submodule in model.children():
            if len(list(submodule.children())) == 0:
                handle = submodule.register_forward_hook(self.hook_fn)
                self.handles.append(handle)
            else:
                self.register(submodule)

    def hook_fn(self, module, input, output):
        # Save the intermediate output
        self.outputs.append(output)
        self.inputs.append(input)
        self.modules.append(module)

    def remove_hooks(self):
        # Remove all the registered hooks
        for handle in self.handles:
            handle.remove()


def trace_weights(hook):
    mapping = {}
    layer_id = 0
    for layer in zip(hook.inputs, hook.outputs, hook.modules):
        layer_id += 1

        if isinstance(layer[2], torch.nn.Conv2d):
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
            # for j in range(h):
            #     for k in range(w):
            #         output_position = (i, j, k)
            #         bias = 0
            #         # Calculate offsets on the input
            #         h_offset = j * stride[0]
            #         w_offset = k * stride[1]
            #         which_group = int(i / output_per_group) * input_per_group
            #         input_positions = []
            #         map_weights = []
            #         # for q in range(input_per_group):
            #         #     for m in range(kernel_size[0]):
            #         #         for n in range(kernel_size[1]):
            #         #             input_positions.append((
            #         #                 which_group * input_per_group + q, h_offset + m - padding[0],
            #         #                 w_offset + n - padding[1]))
            #         #             map_weights.append(weights[i, q, m, n].tolist())
            #         mapping[f"c_{layer_id}_{output_position}_{bias}"] = {
            #             "i": [[which_group], [input_per_group], [h_offset, w_offset]], "w": [i]}
            #         conv_weight[f"{layer_id}_{i}"] = weights[i, :, :, :].tolist()

        if isinstance(layer[2], torch.nn.Linear):
            b_in, c_in = layer[0][0].shape
            b_out, c_out = layer[1].shape
            weights = layer[2].weight.detach().tolist()
            info = {
                "b_in": b_in,
                "c_in": c_in,
                "b_out": b_out,
                "c_out": c_out,
            }
            bias = layer[2].bias.detach().tolist()
            # for i in range(b_out):
            #     for j in range(c_out):
            #         output_position = (i, j)
            #         input_positions = []
            #         map_weights = []
            #         bias = float(layer[2].bias[j].detach())
            #         for m in range(c_in):
            #             input_positions.append((i, m))
            #             map_weights.append(layer[2].weight[j, m].detach().numpy().tolist())
            mapping[f"{layer_id}"] = {"Linear": {"w": weights, "info": info, "bias": bias}}

            # linear_output = layer[1][0].flatten().detach().numpy()
            # file_path = "linear_output.txt"
            # np.savetxt(file_path, linear_output)
            # linear_input = layer[0][0].detach().numpy()
            # file_path = "linear_input.txt"
            # np.savetxt(file_path, linear_input)

        if isinstance(layer[2], torch.nn.BatchNorm2d):
            weights = layer[2].weight.detach().tolist()
            bias = layer[2].bias.detach().tolist()
            r_m = layer[2].running_mean.detach().tolist()
            r_v = layer[2].running_var.detach().tolist()
            input_shape = layer[0][0].shape
            output = layer[2](layer[0][0]);
            mapping[f"{layer_id}"] = {
                "BatchNorm2d": {"w": weights, "bias": bias, "r_m": r_m, "r_v": r_v, "input_shape": input_shape}}

        if isinstance(layer[2], torch.nn.ReLU6):
            input_shape = layer[0][0].shape
            mapping[f"{layer_id}"] = {"ReLU6": {"input_shape": input_shape}}
        
        #TODO what else do we need to do max pooling
        if isinstance(layer[2],nn.MaxPool2d):
            kernel_size = layer[2].kernel_size
            stride = layer[2].stride
            dilation = layer[2].dilation
            (n, c, h, w) = layer[0][0].shape
            (c1, h1, w1) = layer[1][0].shape
            mapping[f"{layer_id}"] = {
                "MaxPool2d": {"s": stride, "k": kernel_size, "d": dilation}
            }

        if layer_id == 22 :
            # output = layer[2](layer[0][0])
            # np.savetxt("../test_references/141.txt", layer[1][0].flatten().detach().numpy(), fmt='%.10f', delimiter=',')
            break
        print(f"layer {layer_id} finished")
    return mapping


# Load the pretrained AlexNet model
model = alexnet(weights="DEFAULT")
model.eval()

#Replace the AdaptiveAvgPool layer with AvgPool
model.avgpool = torch.nn.AvgPool2d(1,1)
#switch out ReLu with ReLu6
replace_relu_mod(model)

# Instantiate the hook
hook = IntermediateOutputsHook()
hook.register(model)
input_image = Image.open("../images/img2.png")
input_image = input_image.convert("RGB")
# print("Image Mode:", input_image.mode)
preprocess = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
print(input_tensor)
input_batch = input_tensor.unsqueeze(0)
# input_data = torch.rand((1, 3, 44, 44))
# Forward pass with the hooked model
output = model(input_batch)
max_scores, predicted_classes = torch.max(output, dim=1)

# 'predicted_classes' now contains the predicted class indices for each sample
print("Predicted classes:", predicted_classes)

# You might want to convert it to a Python list if needed
predicted_classes_list = predicted_classes.tolist()
print("Predicted classes (as list):", predicted_classes_list)

# You can also get the maximum score for each sample
print("Max scores:", max_scores)
# Access the intermediate outputs
intermediate_outputs = hook.outputs
mapping = trace_weights(hook)
with open('../json_files/alexnet.json', 'w') as file:
    json.dump(mapping, file)
print("-----")
# Remove the hooks after you're done
hook.remove_hooks()
