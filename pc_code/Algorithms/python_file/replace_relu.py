import torch
#Replace the ReLU layers with ReLU6 - as ReLU6 is already implemented by the framework and is a lighter option than ReLU
#Based on https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/10 and https://stackoverflow.com/questions/58297197/how-to-change-activation-layer-in-pytorch-pretrained-module/64161690#64161690
def replace_relu_attr(module):
    for attributes in dir(module):
        curr_attribute = getattr(module, attributes)
        if type(curr_attribute) == torch.nn.ReLU:
            new_activation = torch.nn.ReLU6(curr_attribute.inplace)
            setattr(module, attributes, new_activation)
    for _, child in module.named_children():
            replace_relu_attr(child)

def replace_relu_mod(module):
    for child in module.named_children():
        if type(child[1]) == torch.nn.ReLU:
            setattr(module,child[0], torch.nn.ReLU6(True))
        else:
            replace_relu_mod(child[1])
