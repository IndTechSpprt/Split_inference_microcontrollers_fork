def fuse_conv_and_bn(conv, bn):
	#
	# init
	fusedconv = torch.nn.Conv2d(
		conv.in_channels,
		conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		bias=True
	)
	#
	# prepare filters
	w_conv = conv.weight.clone().view(conv.out_channels, -1)
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
	fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
	#
	# prepare spatial bias
	if conv.bias is not None:
		b_conv = conv.bias
	else:
		b_conv = torch.zeros( conv.weight.size(0) )
	b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	fusedconv.bias.copy_( torch.matmul(w_bn, b_conv) + b_bn )
	#
	# we're done
	return fusedconv

import json
import torch
torch.set_grad_enabled(False)
from torchvision.models import resnet18
import numpy as np

# Load the pretrained ResNet model
model = resnet18(weights="DEFAULT")
model.eval()

unfused = torch.nn.Sequential(model.conv1, model.bn1)

fusedconv = fuse_conv_and_bn(model.conv1, model.bn1)

# Custom input tensor - dimensions same as `test_convolution()` in `pc_code/Algorithms/src/main.rs`
input_data = torch.zeros((1, 3, 44, 44))

# Populate the tensor with the desired values (Rust code does the same thing)
for c in range(3):
    for i in range(44):
        input_data[0, c, i, :] = torch.tensor([float(i) for _ in range(44)], dtype=torch.float64)

out = fusedconv.forward(input_data)
out_unmerged = unfused.forward(input_data)

np.savetxt("../test_references/test_resnet18_merged_out.txt",out.flatten().detach().numpy(), fmt='%.10f', delimiter=',')
np.savetxt("../test_references/test_resnet18_unmerged_out.txt",out_unmerged.flatten().detach().numpy(), fmt='%.10f', delimiter=',')
print("-----")