import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule
from torchvision.utils import save_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--classifier_free', action='store_true', default=False, help='train without classifier (default: False)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def sample_and_save_images(n_images, img_size, diffusor, model, reverse_transform, device, store_path):
    # TODO: Implement - adapt code and method signature as needed
    
    # sample images from the model
    imgs_at_t = diffusor.sample(model, img_size, n_images)
    # reverse the transformation applied to the images
    imgs_at_t = reverse_transform(imgs_at_t) 

    return imgs_at_t


def test(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    batch_size = args.batch_size
    timesteps = args.timesteps

    with torch.no_grad():
        pbar = tqdm(testloader)
        for step, (images, labels) in enumerate(pbar):

            images = images.to(device)

            # Algorithm 1 line 3: sample t uniformly for every example in the batch
            t = torch.randint(0, timesteps, (len(images),), device=device).long()
            loss = diffusor.p_losses(model, images, labels, t, loss_type="l2")

            if step % args.log_interval == 0:
                print('Test Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    step, step * len(images), len(testloader.dataset),
                    100. * step / len(testloader), loss.item()))


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, labels, t, loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


# def test(args):
#     # TODO (2.2): implement testing functionality, including generation of stored images.
#     pass


def run(args):
    timesteps = args.timesteps
    classifier_free_guidance = args.classifier_free
    image_size = 32  # TODO: (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size,
                 channels=channels,
                 dim_mults=(1, 2, 4,),
                 class_free_guidance=classifier_free_guidance).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_scheduler = lambda time_steps: linear_beta_schedule(0.0001, 0.02, time_steps)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255 to get values in [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to meet model input requirements
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2), # scale data back to [0, 1]
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.), # scale data back to [0, 255]
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        test(model, valloader, diffusor, device, args)

    test(model, testloader, diffusor, device, args)

    save_path = "<path/to/my/images>"  # TODO: Adapt to your needs
    n_images = 8
    sample_and_save_images(n_images, diffusor, model, reverse_transform, device, save_path)
    torch.save(model.state_dict(), os.path.join("/proj/aimi-adl/models", args.run_name, f"ckpt.pt"))


if __name__ == '__main__':
    args = parse_args()
    # TODO: (2.2): Add visualization capabilities
    run(args)
