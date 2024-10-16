import torch
from torchvision.models import resnet18, alexnet
import torch.nn as nn

# Load tensors from a file
input = torch.load("adap_in.pt", weights_only=True)
expected = torch.load("adap_out.pt", weights_only=True)

m = nn.AvgPool2d(1,1)
out = m(input)
out = torch.squeeze(out,0)

if torch.equal(out, expected):
    print("NICE")