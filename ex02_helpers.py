import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
from torch import nn, from_numpy, tensor
from inspect import isfunction
from einops.layers.torch import Rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def not_exist_create(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


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


def  visualize_forward_diffusion(diffusor_list: list, schedulers: list, img: list, cols: int, time_step: int, return_fig: bool = False):
    """visualize different stages of beta-scheduling during forward diffusion
    
    Parameters
    ----------
    diffusors : list
        list of diffusors with different beta-scheduling
    schedulers : list(str)
        list of beta-scheduler names for each diffusor
    img : np.ndarray, shape: (height, width, channels)
        image to be diffused within range [0, 1], don't need to be scaled to [-1, 1] because we don't use Unet in forward diffusion
    cols : int
        number of columns in the grid
    time_step : int
        current time step
    return_fig : bool, optional
        whether to return the figure or not, by default False

    Returns
    -------
    grid: numpy.ndarray
        grid of images
    """
    # convert the img to tensor
    img = from_numpy(img).permute(2, 0, 1).unsqueeze(0) # shape: (1, channels, height, width)

    noised_imgs = []
    for i, diffusor in enumerate(diffusor_list):
        noised_imgs.append(diffusor.q_sample(img, tensor([time_step,])).squeeze(0).permute(1, 2, 0).numpy())

    rows = len(noised_imgs) // cols
    # let's first create a grid of images with 5 pixels margin in between each image, 50 pixels margin from the top.
    # we also want to append 20 pixels along the bottom of each row to write names of the beta schedulers, so we add 20 pixels
    # to the height of the grid for each row
    grid = np.ones((rows * noised_imgs[0].shape[0] + 5 * (rows - 1) + 70,
                    cols * noised_imgs[0].shape[1] + 5 * (cols - 1) + 40,
                    3))
    for i, im in enumerate(noised_imgs):
        row = i // cols
        col = i % cols
        
        # add 20 pixels white margin along the bottom of the im to write the name of beta scheduler
        img = np.concatenate((im, np.ones((20, im.shape[1], 3))), axis=0)
        img = Image.fromarray(np.uint8(img * 255))
        draw = ImageDraw.Draw(img)
        draw.text((img.size[0] // 2 - 70, img.size[1] - 13), f"beta: {schedulers[i]}", (0, 0, 0))
        # convert the img back to numpy array between [0, 1]
        im = np.array(img) / 255.
        # let's now put the image in the grid
        grid[row * (im.shape[0] + 5) + 50 : (row + 1) * im.shape[0] + 5 * row + 50,
             col * (im.shape[1] + 5) + 20 : (col + 1) * im.shape[1] + 5 * col + 20, :] = im

    # Now we have to add the time-step to the grid
    img = Image.fromarray(np.uint8(grid * 255))
    draw = ImageDraw.Draw(img)
    draw.text((img.size[0] // 2 - 70, 25), f"time-step: {time_step}", (0, 0, 0))
    grid = np.array(img) # grid.dtype = np.uint8 because PIL operates on uint8 images with values in the range [0, 255]

    if return_fig:
        return grid
    else:
        plt.axis("off")
        plt.imshow(grid)
        plt.show()


def visualize_reverse_diffusion(imgs: list, cols: int, epoch: int, time_step: int, return_fig: bool = False):
    """visualize: visualize a list of images in a grid

    Parameters
    ----------
    imgs : list(np.uint8)
        list of images of shape (height, width, channels), dtype: np.uint8, range: [0, 255]
    cols : int
        number of columns in the grid
    epoch : int
        current epoch
    time_step : int
        current time step
    return_fig : bool, optional
        whether to return the figure or not, by default False

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure containing the grid of images
    """
    rows = len(imgs) // cols
    # multiply the grid by 255 to make it in the range [0, 255], 
    grid = np.ones((rows * imgs[0].shape[0] + 5 * (rows - 1) + 70,
                    cols * imgs[0].shape[1] + 5 * (cols - 1) + 40,
                    3), dtype=np.uint8) * 255
    for i, im in enumerate(imgs):
        row = i // cols
        col = i % cols
        grid[row * (im.shape[0] + 5) + 50 : (row + 1) * im.shape[0] + 5 * row + 50,
             col * (im.shape[1] + 5) + 20 : (col + 1) * im.shape[1] + 5 * col + 20, :] = im

    # add text to the image in the horizontol center with 25 pixels margin from top using PIL
    # we multiply the grid by 255 because PIL expects values in the range [0, 255], but our grid is in the range [0, 1]
    img = Image.fromarray(grid)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype("arial.ttf", 162)
    draw.text((img.size[0] // 2 - 70, 25), f"epoch: {epoch}, time-step: {time_step}", (0, 0, 0))
    grid = np.array(img) # grid.dtype = np.uint8 because PIL operates on uint8 images with values in the range [0, 255]

    if return_fig:
        return grid
    else:
        plt.axis("off")
        plt.imshow(grid)
        plt.show()


def create_video(imgs: list, epoch: int, fps: str, store_path: str):
    """create_video: create a video from a list of images

    Parameters
    ----------
    imgs : list
        list of images to create a video from
    epoch : int
        current epoch
    fps : int
        current time step
    store_path : str
        path to store the video
    """
    # convert the images from RGB to BGR because cv2.VideoWriter expects BGR images
    imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]

    # check whether if already video exist for previous epochs, if so, append the new images to the video
    if epoch == 0 or epoch == None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(store_path, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
        for img in imgs:
            video.write(img)
        video.release()
        return
    else:
        video = cv2.VideoCapture(store_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        prev_frames = []
        while True:
            ret, frame = video.read() # frame is in BGR format
            if ret:
                prev_frames.append(frame)
            else:
                break
        video.release()
        prev_frames.extend(imgs)

        video_new = cv2.VideoWriter(store_path, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
        for frame in prev_frames:
            video_new.write(frame)
        video_new.release()
        

if __name__ == "__main__":
    imgs_at_t = [np.random.rand(64, 64, 3) for _ in range(32)] # batch of images at timestep t
    # visualize(imgs_at_t, 8, 1, 1)
    experiment_name = "peanut-butter-fries"
    epochs = 20
    fps = 120
    for e in range(epochs):
        timesteps_imgs = [visualize_reverse_diffusion(imgs_at_t, 8, e, t, return_fig=True) for t in range(0, 500)]
        create_video(timesteps_imgs, e, fps, f"videos/video_{experiment_name}.mp4")
        print(f"epoch {e} done")
