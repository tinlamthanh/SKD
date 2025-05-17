import torch
import torch.nn as nn

from .registry import LOSS

@LOSS.register()
class MSE:
  """
  Mean Squared Error loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.mse_loss(*args, **kwargs)

@LOSS.register()
class BinaryCrossEntropy:
  """
  Binary Cross Entropy loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.binary_cross_entropy(*args, **kwargs)

@LOSS.register()
class BinaryCrossEntropyWithLogits:
  """
  Binary Cross Entropy with Logits loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.binary_cross_entropy_with_logits(*args, **kwargs)
  
@LOSS.register()
class CrossEntropy:
  """
  Cross Entropy loss.
  """
  def __init__(self, *args, **kwargs):
    pass
  
  def __call__(self, *args, **kwargs):
    return nn.functional.cross_entropy(*args, **kwargs)