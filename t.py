import torch

dataset = torch.load("/home/mux/code_workspace/multi-ae-ace/data/esc50_fixed_dataset.pt")
train = dataset["train"][0]
print((train.shape))
