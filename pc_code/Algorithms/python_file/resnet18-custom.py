import torch
from torchvision.models import resnet18
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

# Conver the input into a mini-batch so the model gets the image in the correct format
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

# Process the outputs so we get a suitable output
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

def init_resnet():
    resnet18_default = resnet18(weights="DEFAULT")
    return resnet18_default.eval()    

def infer(input, model):
    output = model(input)
    return process_outputs(output)

def print_tuple_list(tuple_list):
    for tuple in tuple_list:
        print(str(tuple[0]) + ": " + str(tuple[1]), end=' ')
    print("")

#Init with default weights
resnet18_default = init_resnet()
resnet18_custom = init_resnet()

#Switch out avgpool with custom avg pool
resnet18_custom.avgpool = torch.nn.AvgPool2d((7,7),512)

#prepareinput
prepared_input = prepare_input()

#Run inference
out_default = infer(prepared_input,resnet18_default)
out_custom = infer(prepared_input,resnet18_custom)

#Print
print("Default:", end=' ')
print_tuple_list(out_default)
print("Custom: ",end=' ')
print_tuple_list(out_custom)