import torch
from torchvision.models import resnet18, alexnet
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from replace_relu import replace_relu_attr, replace_relu_mod

# Convert the input into a mini-batch so the model gets the image in the correct format
def prepare_input():
    #cheetah-resize-224/resize/224/00000011_224resized.png from https://www.kaggle.com/datasets/anshulmehtakaggl/wildlife-animals-images
    input_image = Image.open("../images/img2.png")
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0) #Create a mini-batch - required by the model

# Process the inference outputs to get top 5 classes and probabilities
def process_outputs(output):
    probs = torch.nn.functional.softmax(output[0], dim=0)

    #from https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_probs, top5_ids = torch.topk(probs, 5)

    out = []
    for prob_cat_pair in range((top5_probs.size(0))):
        out.append((categories[top5_ids[prob_cat_pair]], top5_probs[prob_cat_pair].item()))
    return out

# Initialize ResNet18 with default weights (IMAGENET1K_V1)
def init_resnet():
    model = resnet18(weights="DEFAULT")
    return model.eval()    

#Initialize AlexNet with default weights (IMAGENET1K_V1)
def init_alexnet():
    model = alexnet(weights="DEFAULT")
    return model.eval()

# Run inference and return results
def infer(input, model):
    output = model(input)
    return process_outputs(output)

# Custom print function for tuples with the prediction class and probability
def print_tuple_list(tuple_list):
    for tuple in tuple_list:
        print(str(tuple[0]) + ": " + str(tuple[1]), end=' ')
    print("")

#Init with default weights
resnet18_default = init_resnet()
resnet18_custom = init_resnet()
alexnet_default = init_alexnet()
alexnet_custom = init_alexnet()

#Switch out avgpool with custom avg pool
resnet18_custom.avgpool = torch.nn.AvgPool2d((7,7),512)
alexnet_custom.avgpool = torch.nn.AvgPool2d(1,1)

#switch out ReLu with ReLu6
replace_relu_attr(resnet18_custom)
replace_relu_mod(alexnet_custom)

#prepareinput
input = prepare_input()

#Run inference
out_resnet_default = infer(input,resnet18_default)
out_resnet_custom = infer(input,resnet18_custom)
out_alexnet_default = infer(input, alexnet_default)
out_alexnet_custom = infer(input, alexnet_custom)

#Print
print("ResNet18")
print("Default:", end=' ')
print_tuple_list(out_resnet_default)
print("Custom: ",end=' ')
print_tuple_list(out_resnet_custom)
print("---------------------------")
print("AlexNet")
print("Default:", end=' ')
print_tuple_list(out_alexnet_default)
print("Custom: ",end=' ')
print_tuple_list(out_alexnet_custom)