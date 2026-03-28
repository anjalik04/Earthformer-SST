import torch
import torch.nn as nn

class BiasCorrectedStudent(nn.Module):
    """
    Wraps a trained student model and adds a fixed bias offset to its predictions.
    No retraining — just post-processing correction.
    """
    def __init__(self, student: nn.Module, bias: float = 0.0):
        super().__init__()
        self.student = student
        self.bias = bias

    def forward(self, x):
        pred = self.student(x)
        return pred + self.bias