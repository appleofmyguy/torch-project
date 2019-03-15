import torch
import torch.nn as nn
import torch.nn.functional as F


class ExampleModel(nn.Module):
    def __init__(self, arg):
        super(ExampleModel, self).__init__()
        self.arg = arg

        self.projection = nn.Linear(10, 5, bias=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        pass

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        raise NotImplementedError("not implemented")
        params = {}  # fill this out
        torch.save(params, path)

