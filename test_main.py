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
from ex02_diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from ex02_helpers import create_video, visualize_reverse_diffusion, not_exist_create, delete_if_exist
from torchvision.utils import save_image


def sample_and_save_images(n_images,
                           labels,
                           img_size,
                           diffusor,
                           model,
                           epoch,
                           reverse_transform,
                           experiment_name,
                           save_video=True,
                           video_fps=120,
                           store_path="./images",
                           sample_images=False):# create seperate video for sample
    # TODO: Implement - adapt code and method signature as needed

    # set model to evaluation mode because classifier-free implementation uses this flag to determine which if-else branch
    # to use during training and evaluation.
    model.eval() 
    
    # sample images from the model
    imgs_at_t = diffusor.sample(model, labels, img_size, n_images) # shape: [[n_images, 3, img_size, img_size], ...], len: timesteps
    # reverse the transformation applied to the images
    # After reverse transformation, we have a list of lists of length timesteps, where each inner list
    # contains n_images of shape: (img_size, img_size, 3), dtype: np.uint8, values in [0, 255].
    imgs_at_t = [[reverse_transform(img) for img in img_batch_t.cpu()] for img_batch_t in imgs_at_t]

    if save_video:
        visualization_t = [
            visualize_reverse_diffusion(img_batch_t, 8, epoch, t, return_fig=True)
            for t, img_batch_t in enumerate(imgs_at_t)
        ]
        not_exist_create(f"{store_path}/{experiment_name}")
        # change epoch to 0 or None if you want to creat a new video
        if sample_images:
            epoch = None
            temp_store_path = f"{store_path}/{experiment_name}/{experiment_name}_sample.mp4"
        else:
            temp_store_path = f"{store_path}/{experiment_name}/{experiment_name}.mp4"
        create_video(visualization_t, epoch, video_fps, temp_store_path) # create directory if not exists

    # store the images at timestep T
    imgs_T = np.asarray(imgs_at_t[-1]) # shape: (n_images, img_size, img_size, 3), dtype: np.uint8, values in [0, 255]
    imgs_T = imgs_T / 255. # scale to [0, 1]
    imgs_T = torch.from_numpy(imgs_T) # shape: (n_images, img_size, img_size, 3), dtype: torch.float64, values in [0, 1]
    imgs_T = imgs_T.permute(0, 3, 1, 2) # shape: (n_images, 3, img_size, img_size)
    # create directory if not exists
    not_exist_create(f"{store_path}/{experiment_name}")
    save_image(imgs_T, f"{store_path}/{experiment_name}/epoch_{epoch}.jpg") # it stores in grid of 8 columns by default

    return imgs_at_t


def test(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    batch_size = args.batch_size
    timesteps = args.timesteps

    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(testloader)
        for step, (images, labels) in enumerate(pbar):

            images = images.to(device)
            labels= labels.to(device)

            # Algorithm 1 line 3: sample t uniformly for every example in the batch
            t = torch.randint(0, timesteps, (len(images),), device=device).long()
            loss = diffusor.p_losses(model, images, labels, t, loss_type="l2")
            total_loss += loss.item()

            if step % args.log_interval == 0:
                print('Test Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    step, step * len(images), len(testloader.dataset),
                    100. * step / len(testloader), loss.item()))
            if args.dry_run:
                return total_loss

    return total_loss


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    model.train() # remember to set model to train mode when training
    pbar = tqdm(trainloader)
    total_loss = 0
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, labels, t, loss_type="l2")
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            return total_loss
        
    return total_loss


# def test(args):
#     # TODO (2.2): implement testing functionality, including generation of stored images.
#     pass


def run(args):
    # read arguments
    timesteps = args.timesteps
    classifier_free_guidance = args.classifier_free
    num_classes = args.num_classes
    sample_images = args.sample_images # Flag to sample images during training
    num_samples = args.num_samples
    save_video = args.save_video
    video_fps = args.video_fps
    experiment_name = args.experiment_name
    store_path = args.store_path
    image_size = 32  # TODO: (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size,
                 channels=channels,
                 dim_mults=(1, 2, 4,),
                 class_free_guidance=classifier_free_guidance,
                 num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # my_scheduler = lambda time_steps: linear_beta_schedule(0.0001, 0.02, time_steps)
    my_scheduler = lambda time_steps: cosine_beta_schedule(time_steps)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, classifier_free_guidance=classifier_free_guidance, device=device)

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
        # ToPILImage(), we don't need this here because we need numpy arrays for visualization and saving
    ])

    dataset = datasets.CIFAR10('./CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('./CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    # let's create labels for classifier-free sampling
    if classifier_free_guidance:
        labels = torch.randint(0, num_classes, (num_samples,), device=device).long()
        # cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        label_2_class = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
                         5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        classes = [label_2_class[label.item()] for label in labels]    
        print(f"labels: {labels}")
        print(f"classes: {classes}")
    else:
        labels = None

    best_loss = np.inf
    best_epoch = 0
    for epoch in range(epochs):
        train_loss = train(model, trainloader, optimizer, diffusor, epoch, device, args)
        val_loss = test(model, valloader, diffusor, device, args)

        if sample_images and (epoch % args.sample_interval == 0):
            # TODO: need to make sure it works with classifier-free guidance
            sample_and_save_images(num_samples,
                                   labels,
                                   image_size,
                                   diffusor,
                                   model,
                                   epoch,
                                   reverse_transform,
                                   experiment_name,
                                   save_video,
                                   video_fps,
                                   store_path)
        
        if val_loss < best_loss:
            not_exist_create(f"./models/{args.experiment_name}")  # create directory if not exists
            delete_if_exist(f"./models/{args.experiment_name}/epoch_{best_epoch}_ckpt.pt") # delete previous best model
            torch.save(model.state_dict(), os.path.join("./models", args.experiment_name, f"epoch_{epoch}_ckpt.pt"))
            best_loss = val_loss
            best_epoch = epoch
            print(f"epoch: {epoch}'s val_loss: {val_loss} is better than previous best loss: {best_loss}. Saving model...")

        # log train and validation loss
        print(f"epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}")
            

    test(model, testloader, diffusor, device, args)

    # sample and store images at the end of training
    sample_and_save_images(num_samples,
                           labels,
                           image_size,
                           diffusor,
                           model,
                           None, # epoch is not valid for inference
                           reverse_transform,
                           experiment_name,
                           save_video,
                           video_fps,
                           store_path,
                           True)
    torch.save(model.state_dict(), os.path.join("./models", args.experiment_name, f"final_ckpt.pt"))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    args = {
        "experiment_name": "gemini-1",
        "dry_run": False,
        "no_cuda": False,
        "timesteps": 500, # usually 1000
        "classifier_free": False,
        "num_classes": 10,
        "epochs": 100,
        "batch_size": 8,
        "lr": 0.0001,
        "log_interval": 1,
        "sample_interval": 1,
        "sample_images": True,
        "num_samples": 8,
        "save_video": True,
        "video_fps": 120,
        "store_path": "./output"
    }
    args = dotdict(args)
    # TODO: (2.2): Add visualization capabilities
    run(args)
