from torch import nn
from inspect import isfunction
from einops.layers.torch import Rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


def extract(a, t, x_shape):
    """extract: extract values of "a" belong the timestep t in the batch

    Parameters
    ----------
    a : (timesteps,)
        one of the parameters during the forward and reverse diffusion process
    t : (batch_size,)
        time step for each sample in the batch
    x_shape : (batch_size, channels, height, width)
        shape of the input image

    Returns
    -------
    (batch_size, 1, 1, 1)
        reshaped values of "a" belong the timestep t in the batch
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()) # get values of "a" belong the timestep t in the batch. shape: (batch_size,)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device) # shape: (batch_size, 1, 1, 1)