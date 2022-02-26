"""
https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/models/torch/torch_modelv2.py
This is for PyTorch but TensorFlow is analogous
"""
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class YourModel(TorchModelV2):
    pass
