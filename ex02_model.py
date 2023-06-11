import math
import random
from functools import partial
from einops import rearrange, reduce
import torch
from torch import nn, einsum
import torch.nn.functional as F
from ex02_helpers import *


# Note: This code employs large parts of the following sources:
# Niels Rogge (nielsr) & Kashif Rasul (kashif): https://huggingface.co/blog/annotated-diffusion (last access: 23.05.2023),
# which is based on
# Phil Wang (lucidrains): https://github.com/lucidrains/denoising-diffusion-pytorch (last access: 23.05.2023)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    # Note: Here we're inheriting from nn.Conv2d because we want to modify the
    # weight initialization, that's why we use F.conv2d(...) to provide the custom
    # filter weights to the forward pass

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

# Here try to think intuitvely about the group-normalization. What does it do?
# you can look at it at: https://theaisummer.com/normalization/
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    # Note: This implements FiLM conditioning, see https://distill.pub/2018/feature-wise-transformations/ and
    # http://arxiv.org/pdf/1709.07871.pdf
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        if exists(time_emb_dim) and exists(classes_emb_dim):
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2) # dim_out*2 because we need to separate scale and shift
            )
        elif exists(time_emb_dim):
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(int(time_emb_dim), dim_out * 2) # dim_out*2 because we need to separate scale and shift
            )
        else:
            self.mlp = None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        # time_emb: (b, dim*4), class_emb: (b, dim_in*4). dim is different from dim_in and dim_out
        # dim is the dimension of the intermediate channels somewhat similar to dimension in the transformer.
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) and exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1) # shape: (b, dim*4 + dim*4). concatenate time_emb and class_emb
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1') # shape: (b, dim_out*2, 1, 1)
            scale_shift = cond_emb.chunk(2, dim=1) # separate scale and shift
        elif exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1') # shape: (b, dim_out*2, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1) # separate scale and shift
        else:
            raise ValueError("Either both time_emb and class_emb should be provided or only time_emb should be provided")

        h = self.block1(x, scale_shift=scale_shift) # x: (b, dim, h, w), scale_shift: ((b, dim_out, 1, 1), (b, dim_out, 1, 1))
        # Here x is (b, dim, h, w) but in Block's forward pass, it is projected to (b, dim_out, h, w). Then we can shift and scale
        h = self.block2(h)

        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False) # 1x1 convolution to map the input to q, k, v
        self.to_out = nn.Conv2d(hidden_dim, dim, 1) # 1x1 convolution to map the output to the original dimension

    def forward(self, x):
        # Here dim_head*heads*3 is encoded in the form of channels. Isn't it interesting?
        b, c, h, w = x.shape
        # .chunk() split the qkv tensor into 3 parts for q, k, v in a tuple
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # we map the tuple to transform tensor of k, q, v into shape (b, heads*dim_head, h, w)
        # rearrange() is used to rearrange to shape (b, heads, dim_head, h*w). h*w is the number of pixels in the image
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        # q, k, v are now of shape (b, heads, dim_head, h*w) # dim_head for each pixel, and total number of pixels is h*w
        # here we compute similiarity between each pixel in the image, however, in ViT
        # we compute similiarity between each token/patch in the sequence
        sim = einsum("b h d i, b h d j -> b h i j", q, k) # sim: (b, heads, h*w, h*w)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach() # Here we normalize the similiarity matrix by subtracting the max value
        attn = sim.softmax(dim=-1) # attn: (b, heads, h*w, h*w)

        # it will automaticaly transpose the v tensor to shape (b, heads, h*w, dim_head)
        out = einsum("b h i j, b h d j -> b h i d", attn, v) # out: (b, heads, h*w, dim_head)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w) # out: (b, heads*dim_head, h, w)
        return self.to_out(out) # out: (b, dim, h, w)


# Linear attention variant, scales linear with sequence length
# Shen et al.: https://arxiv.org/abs/1812.01243
# https://github.com/lucidrains/linear-attention-transformer
# Above we have implemented dot-product attention, here we're going to look a variant of
# attention known as efficient attention, which is considered to be more computationally, and
# specially memory efficient. Mathematically, it is equivalent to the dot-product attention, however,
# it grows linearly with the sequence length, instead of quadratically. For details, please refer to
# arxiv paper that explains how the projection/softmax is differently computed in this case.
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        # Here q, k, v are of shape (b, heads, dim_head, h*w)
        # where h*w is the number of input pixels, and dim_head is the dimension of each pixel
        # To better understand the dim=-2, dim=-1 please refer the Figure 1 in the paper
        # mentioned above.
        q = q.softmax(dim=-2) # Here we compute softmax along the second-last dimension
        k = k.softmax(dim=-1) # Here we compute softmax along the last dimension

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)# shape: (b, dim_head, h, w)
        return self.to_out(out)


# Wu et al.: https://arxiv.org/abs/1803.08494
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim) # apply group normalization along the channel dimension considering
        # the all channels as single group (equivalent to layer normalization), however, we can also group the 
        # channels into 2 or more groups, and apply group normalization along each group. Remember, the output
        # is going to be of same shape as the input regardless of the number of groups we choose.
        # For more details, please refer to the following link: https://theaisummer.com/normalization/
        # (h, w) along perpendicular z-axis, number of channels along y-axis, and batch size along x-axis

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# TODO: make yourself familiar with the code that is presented here, as it closely interacts with the rest of the exercise.
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=4,
        class_free_guidance=False,  # TODO: Incorporate in your code
        num_classes=None,
        p_uncond=None,
    ):
        """Unet

        Parameters
        ----------
        dim : int
            dimension of the intermediate channels
        init_dim : int, optional
            projects the input image to this dimension, by default None
        out_dim : int, optional
            number of channels in the output image, by default None (i.e. same as input)
        dim_mults : tuple, optional
           determines the number of channels in each block of the Unet, by default (1, 2, 4, 8)
        channels : int, optional
            number of channels in the input image, by default 3
        resnet_block_groups : int, optional
            number of groups in each ResNet block for GroupNorm, by default 4
        class_free_guidance : bool, optional
            whether to use class-free guidance, by default False
        num_classes : int, optional
            number of classes in the dataset, by default None
        p_uncond : float e ~ (0.1, 0.2), optional
            probability of replacing the class embedding with the null token, by default None
        """
        super().__init__()

        # determine dimensions
        self.channels = channels
        input_channels = channels   # adapted from the original source
        self.num_classes = num_classes
        self.p_uncond = default(p_uncond, 0.1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)  # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # dims = [init_dim, dim, dim*2, dim*4, dim*8].
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        # dim * 4 has nothing to do with the number of channels in the input image, it is just chosen arbitrarily in the
        # original implementation.
        time_dim = dim * 4 

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        ) # shape: (b, dim*4)

        # TODO: Implement a class embedder for the conditional part of the classifier-free guidance & define a default
        self.class_free_guidance = class_free_guidance
        class_dim = dim*4 if self.class_free_guidance else None
        if self.class_free_guidance:
            # we decided to use the dictionary size of num_classes+1 to account for the null class.
            # The implementation by lucidrains uses the dictionary size of num_classes, and then uses the
            # nn.Parameter() to add the null class embedding. However, we decided to resize the dictionary.
            self.class_embedding = nn.Embedding(self.num_classes+1, dim) # shape: (num_classes+null_class, dim*4)
            self.class_mlp = nn.Sequential(
                nn.Linear(dim, class_dim),
                nn.GELU(),
                nn.Linear(class_dim, class_dim),
            ) # shape: (b, dim*4)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # TODO: Adapt all blocks accordingly such that they can accommodate a class embedding as well
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=class_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=class_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))), # Residual is in helpers.py. It simpley creates a residual connection.
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=class_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim))) # Notice here we're using Attention instead of LinearAttention
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=class_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=class_dim), # dim_out + dim_in because we're concatenating the skip connection
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=class_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        # out_dim is the same number of channels and dimensions as the input image because Unet is predicting
        # the noise that is added to the input image which will be used to calculate the mean in the reverse
        # diffusion process.
        self.out_dim = default(out_dim, channels) # out_dim is the number of channels in the output image

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=class_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1) # 

    def forward(self, x, time, class_cond=None):
        """Unet forward pass

        Parameters
        ----------
        x : (b, channels, h, w) tensor
            batch of images
        time : (b,) int tensor
            time step for each sample in the batch
        class_cond : (b,) int tensor, optional
            class label for each sample in the batch, by default None

        Returns
        -------
        (b, channels, h, w) tensor
            predicted noise for all pixels across all channels
        """
        b = x.shape[0]
        x = self.init_conv(x)
        r = x.clone() # clone here for the final residual connection

        t = self.time_mlp(time) # shape: (b, dim*4)

        # TODO: Implement the class conditioning. Keep in mind that
        #  - for each element in the batch, the class embedding is replaced with the null token with a certain probability during training
        #  - during testing, you need to have control over whether the conditioning is applied or not
        #  - analogously to the time embedding, the class embedding is provided in every ResNet block as additional conditioning
        
        #shape of class_cond: (b,)
        if self.class_free_guidance:
            if self.training:
                # During training, replace the class embedding with the null class with probability p_uncond
                # generate random numbers between 0 and 1 for each element in the batch and then
                # check if the random number is less than p_uncond. If yes, replace the class embedding
                # with the null class embedding. So we replace 20% of the class embeddings with the null class.
                mask = torch.rand(class_cond.shape[0]) < self.p_uncond # shape: (b,)
                class_cond[mask] = self.num_classes # shape: (b,)
                class_embedding = self.class_embedding(class_cond) # shape: (b, dim)
                class_cond = self.class_mlp(class_embedding) # shape: (b, dim*4)
            else:
                # During inference time, we can use class labels if provided otherwise we can use the null class
                if class_cond is None:
                    # create token for null class
                    class_cond = torch.ones(b, dtype=torch.long)*self.num_classes # shape: (b,)
                class_embedding = self.class_embedding(class_cond) # shape: (b, dim)
                class_cond = self.class_mlp(class_embedding) # shape: (b, dim*4)
        

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, class_cond)
            h.append(x)

            x = block2(x, t, class_cond)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, class_cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, class_cond)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, class_cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, class_cond)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1) # r is the residual connection that we stored earlier

        x = self.final_res_block(x, t, class_cond)
        return self.final_conv(x) # shape: (b, 3, h, w) is the predicted noise


if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32) # shape: (b, c, h, w)
    time = torch.randint(0, 4000, (4,)) # shape: (b,)
    class_cond = torch.randint(0, 10, (4,)) # shape: (b,)
    unet1 =  Unet(32, channels=3, class_free_guidance=True, num_classes=10, p_uncond=0.1)
    print(unet1(x, time, class_cond).shape)
    unet2 = Unet(32, channels=3)
    print(unet2(x, time).shape)
