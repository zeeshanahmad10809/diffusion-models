import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm
from functools import partial

# TODO: remove this import from here
from ex02_model import Unet


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper

    Parameters
    ----------
    beta_start : float
        starting value for beta
    beta_end : float
        end value for beta
    timesteps : int
        number of timesteps

    Returns
    -------
    torch.Tensor
        beta schedule
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO: (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    T = timesteps
    t = torch.arange(0, T+1, dtype=torch.float32)

    # compute f_t
    t_over_T_plus_s = t / T + s
    one_plus_s = 1 + s
    f_t = torch.cos((t_over_T_plus_s / one_plus_s) * (torch.pi / 2))**2

    # compute alpha_bar_t
    alpha_bar_t = f_t / f_t[0]

    # compute beta_t
    alpha_bar_t_minus_1 = alpha_bar_t[:-1] # make sure to first slice alpha_bar_t_minus_1 before slicing alpha_bar_t
    alpha_bar_t = alpha_bar_t[1:]
    beta_t = 1 - (alpha_bar_t / alpha_bar_t_minus_1)
    # IDDPM paper recommends to clip beta_t to 0.999, but don't provide information regarding the lower bound.
    # But for linear beta schedule, they use 0.0001 as lower bound, so we use the same lower bound for cosine beta schedule.
    beta_t.clip_(min=0.0001, max=0.999)

    return beta_t


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO: (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    slimit = (-6, 6)
    T = timesteps
    t = torch.arange(0, T+1, dtype=torch.float32)

    # compute input of sigmoid function z_t
    z_t = slimit[0] + (2 * t / T) * slimit[1]

    # compute beta_t
    beta_t = beta_start + torch.sigmoid(z_t) * (beta_end - beta_start)

    return beta_t


class Diffusion:

    # TODO: (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, classifier_free_guidance=False, w=0.3, device="cuda"):
        """Diffusion Model

        Parameters
        ----------
        timesteps : int
            number of timesteps for the forward and reverse diffusion process
        get_noise_schedule : method
            takes timesteps as input and returns a noise tensor of shape (timesteps,)
        img_size : int
            image size (assuming square images)
        classifier_free_guidance : bool, optional
            whether to use classifier-free guidance, by default False
        w : float, optional
            guidace factor, paper shows w ~ [0, 4.0], but works well with w=0.3 for FID, and w=4.0 for IS, by default 0.3
        device : str, optional
            device to use for processing, by default "cuda"
        """
        self.timesteps = timesteps
        self.img_size = img_size
        self.classifier_free_guidance = classifier_free_guidance
        self.w = w
        self.device = device

        # Note: In constructor, we're just calculating the required hyperparameters for forward and reverse diffusion process.
        # All of these hyperparaters are defined in the DDPM paper.
        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps,) # shape (timesteps,)

        # define alphas
        self.alphas = 1 - self.betas # shape (timesteps,)
        self.alphas_bar = torch.cumprod(self.alphas, axis=0) # shape (timesteps,). CAUTION: don't work without axis=o even though it's a 1D tensor
        
        # calculations for diffusion q(x_t | x_{t-1}) and also posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_alpha_bar = torch.sqrt(self.alphas_bar) # shape (timesteps,)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_bar) # shape (timesteps,)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # Note: we prepend 0 because we need t=0 inorder to calculate the posterior for t=1.
        self.alphas_bar_minus_1 = torch.cat((torch.tensor([1]), self.alphas_bar[:-1])) # prepend 0 for t=1. shape (timesteps+1,)
        self.sqrt_recip_alphas = 1 / torch.sqrt(self.alphas) # shape (timesteps,)

    @torch.no_grad()
    def p_sample(self, model, x, label, t, t_index):
        """reverse diffusion process

        Parameters
        ----------
        model : UNet
            noise predictor
        x : (batch_size, channels, img_size, img_size) tensor
            image at timestep t
        label : (batch_size,) tensor
            labels for each sample in the batch
        t : (batch_size,) tensor
            randomly sampled timesteps
        t_index : int
            index of the current timestep

        Returns
        -------
        posterior_mean : (batch_size, channels, img_size, img_size) tensor
            mean image of the posterior distribution q(x_{t-1} | x_t, x_0)
        """
        # TODO: (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        # TODO: (2.2): The method should return the image at timestep t-1.

        # check whether labels is Tensor if not convert it to Tensor
        if label is not None and not torch.is_tensor(label):
            label = torch.tensor(label, device=self.device).long()


        # Here we are going to predict the noise for the timestep t-1 using the model.
        # Then to sample from the posterior distribution q(x_{t-1} | x_t, x_0) we need to
        # sample from N(0, 1) and then scale and shift it using parameteres of posterior std and mean respectively.
        # Remember: In IDDPM paper, we have equ. 10 and equ. 13 that can give us the posterior std and mean respectively.
        # std. have a closed-form solution but mean needs to be estimated using the reparameterization trick and predicting
        # the noise at the timestep t-1 using the model. (We don't directly go from x_t to x_1, but sequentially.)
        # Notice: equ. 13 is obtained by solving equ. 9 and equ. 11.


        # Step-01: Compute variance of the posterior distribution q(x_{t-1} | x_t, x_0) anaytically using equ. 10 in the paper.
        posterior_betas = self.betas * (1 - self.alphas_bar_minus_1) / (1 - self.alphas_bar) # shape (timesteps,)

        # Step-02: Compute the mean of the posterior distribution q(x_{t-1} | x_t, x_0) using the reparameterization trick and equ. 13 in the paper.
        # Step-02.1: compute noise with class-condition and without class-condition
        if self.classifier_free_guidance:
            conditioned_noise = model(x, t, class_cond=label)
            unconditioned_noise = model(x, t)
            # Step-02.2: compute the noise using interpolation between conditioned and unconditioned noise
            pred_noise = (1 + self.w) * conditioned_noise - self.w * unconditioned_noise
        else:
            pred_noise = model(x, t)
        
        posterior_mean = extract(self.sqrt_recip_alphas, t, x.shape) *\
            (x - (extract(self.betas, t, x.shape) * pred_noise) /\
            extract(self.sqrt_one_minus_alpha_bar, t, x.shape)) # shape: (batch_size, channels, img_size, img_size)
        # we could have create self.sqrt_one_minus_alpha_bar and divide it instead of multiplying with self.sqrt_recip_one_minus_alphas_bar
        # it means we're dumb and we don't know how to optimize our code. But we're not dumb, we're smart.

        # Step-03: Sample from the posterior distribution q(x_{t-1} | x_t, x_0) using the reparameterization trick.
        if t_index == 0:
            # simply return the mean of the posterior distribution q(x_{t-1} | x_t, x_0) for t=0. Don't sample.
            return posterior_mean # shape: (batch_size, channels, img_size, img_size)
        else:
            # Here, we sample by scale and shift of standard normal distribution N(0, 1) using the mean and std of the
            # posterior distribution q(x_{t-1} | x_t, x_0), and return the sample x_{t-1} ~ q(x_{t-1} | x_t, x_0).
            posterior_beta_sqrt = torch.sqrt(extract(posterior_betas, t, x.shape)) # sqrt(variance or beta) is std. of the posterior distribution q(x_{t-1} | x_t, x_0)
            return posterior_mean + posterior_beta_sqrt * torch.randn_like(x, device=self.device) # shape: (batch_size, channels, img_size, img_size)

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, labels, image_size, batch_size=16, channels=3):
        """sample from the model

        Parameters
        ----------
        model : UNet
            noise predictor
        image_size : int
            image size (assuming square images)
        labels : (batch_size,) tensor
            labels for each sample in the batch
        batch_size : int, optional
            batch size, by default 16
        channels : int, optional
            number of channels, by default 3

        Returns
        -------
        imgs_at_t : list of (batch_size, channels, img_size, img_size) tensors
            list of images at each timestep
        """
        # TODO: (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.
        # TODO: (2.2): Return the generated images

        # Notice: We here in step-02 iterate back over all of the T timesteps, but in the step-02.1: we sample
        # a random timestep t ~ U(0, T) for each sample in the batch.
        # Maybe this randomness avoids the Unet to overfit to the sequence of images and instead it learns to
        # generate noise for each timestep independently.

        # Step-01: Sample a random noise vector z_T ~ N(0, 1) of shape (batch_size, channels, img_size, img_size)
        x_T = torch.randn(batch_size, channels, image_size, image_size, device=self.device)

        # Step-02: Iterate over all timesteps in the reverse diffusion process
        x_t = x_T
        imgs_at_t = []
        for step_t in tqdm(reversed(range(self.timesteps)), desc="reverse diffusion process", total=self.timesteps):
            # Step-02.1: create a batch of same timestep t for all samples in the batch
            t = torch.tensor([step_t] * batch_size, device=self.device).long() # no need to use long() here, but just to be sure
            x_t = self.p_sample(model, x_t, labels, t, t_index=step_t) # shape: (batch_size, channels, img_size, img_size)
            imgs_at_t.append(x_t)

        # Step-03: Return the generated images
        return imgs_at_t # shape: [(batch_size, channels, img_size, img_size), ...], len(imgs_at_t) = timesteps

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        """forward diffusion process

        Parameters
        ----------
        x_zero : (batch_size, channels, img_size, img_size) tensor
            image at t=0
        t : (batch_size,) tensor
            t ~ U(0, T) where T is the number of timesteps
        noise : (batch_size, channels, img_size, img_size) tensor, optional
             standard normal noise vector, by default None

        Returns
        -------
        x_t : (batch_size, channels, img_size, img_size) tensor
            noised images at timestep t
        """
        # TODO: (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        # Notice we sample the noise vector from a standard normal distribtuion of the same lenght as the image vector
        # and we don't add a single noise constant value because may just shift the color but don't change the distribution
        # of the image.
        if noise is None:
            noise = torch.randn_like(x_zero, device=self.device)

        # extract method extracts the value of forward and diffusion parameters for a particular timestep
        # provided in the batch. For example, if alpha = [0.1, 0.2, 0.3, 0.4], alpha.shape: (timesteps,)
        # t = [1, 2], t.shape: (batch_size,), then extract(alpha, t, x_zero.shape) will return [0.2, 0.3],
        # however, it will reshape the [0.2, 0.3] it to (batch_size, 1, 1, 1) to match the shape of x_zero.
        # So the output will be [[[[0.2]]], [[[0.3]]]].
        # Notice, we have only a constant value for each image and not separate value for each channel or pixel.
        # but how does it destroy the image by adding only a constant value to it over time?
        sqrt_alpha_bar_t_prod_x_zero = extract(self.sqrt_alpha_bar, t, x_zero.shape) * x_zero
        sqrt_one_minus_alpha_bar_t_prod_noise = extract(self.sqrt_one_minus_alpha_bar, t, noise.shape) * noise
        x_t = sqrt_alpha_bar_t_prod_x_zero + sqrt_one_minus_alpha_bar_t_prod_noise
        return x_t

    def p_losses(self, denoise_model, x_zero, label, t, noise=None, loss_type="l1"):
        """
        compute the loss for the reverse diffusion process

        Parameters
        ----------
        denoise_model : UNet
            denoising model
        x_zero : (batch_size, channels, img_size, img_size) tensor
            image at t=0
        label : (batch_size,) tensor
            labels for each sample in the batch
        t : (batch_size,) tensor
            t ~ U(0, T) where T is the number of timesteps
        noise : (batch_size, channels, img_size, img_size) tensor, optional
                standard normal noise vector, by default None
        loss_type : str, optional
            loss type, by default "l1"

        Returns
        -------
        loss : float
            loss value
        """
        # TODO: (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero, device=self.device) # shape: (batch_size, channels, img_size, img_size), channels=3

        # compute x_t using forward diffusion process
        x_t = self.q_sample(x_zero, t, noise) # shape: (batch_size, channels, img_size, img_size)

        # predict the noise at timestep t using the model
        # Important: During training, we just denoise the image at timestep t using the denoising model, we don't need to
        # perform the reverse diffusion process to get the image at timestep t-1.
        if self.classifier_free_guidance:
            noise_pred = denoise_model(x_t, t, class_cond=label)
        else:
            noise_pred = denoise_model(x_t, t) # shape: (batch_size, channels, img_size, img_size), channels=3

        if loss_type == 'l1':
            # TODO: (2.2): implement an L1 loss for this task
            loss = F.l1_loss(noise_pred, noise)
        elif loss_type == 'l2':
            # TODO: (2.2): implement an L2 loss for this task
            loss = F.mse_loss(noise_pred, noise)
        else:
            raise NotImplementedError()

        return loss


if __name__ == "__main__":
    # beta_start = 0.0001
    # beta_end = 0.02
    # scheduler = partial(linear_beta_schedule, beta_start, beta_end)
# 
    # img = torch.randn(1, 3, 32, 32)
    # labels = torch.tensor([3])
    # diffusion = Diffusion(50, scheduler, 32, classifier_free_guidance=True, w=0.3, device="cpu")
    # x = diffusion.q_sample(img, torch.tensor([10]))
    # print(x.shape)
# 
    # unet1 =  Unet(32, channels=3, class_free_guidance=True, num_classes=10, p_uncond=0.1)
    # unet1.eval() # make sure to use eval mode, because we distinguish between train and eval mode in UNet for class-free guidance
    # imgs_at_t = diffusion.sample(unet1, labels, 32, batch_size=1, channels=3)
    # print(len(imgs_at_t))
# 
    # diffusion1 = Diffusion(50, scheduler, 32, device="cpu")
    # x1 = diffusion1.q_sample(img, torch.tensor([10]))
    # print(x1.shape)
# 
    # unet1 =  Unet(32, channels=3)
    # unet1.eval()
    # imgs_at_t1 = diffusion1.sample(unet1, labels, 32, batch_size=1, channels=3)
    # print(len(imgs_at_t1))

    from PIL import Image
    import numpy as np
    from ex02_helpers import visualize_forward_diffusion, create_video

    schedulers = [partial(linear_beta_schedule, 0.0001, 0.02),
                  partial(cosine_beta_schedule, s=0.008),
                  partial(sigmoid_beta_schedule, 0.0001, 0.02),]
                  # partial(cosine_beta_schedule1, s=0.008)]
    scheduler_names = ["linear", "cosine", "sigmoid"]
    diffusors = [Diffusion(1000, scheduler, 1, device="cpu") for scheduler in schedulers]
    img = Image.open("/home/permute/Downloads/pikachu.jpg")
    # reduce size by factor of 4
    img = img.resize((img.size[0] // 4, img.size[1] // 4))
    img = np.array(img) / 255.
    grid = visualize_forward_diffusion(diffusors, scheduler_names, img, 3, 2, return_fig=True)
    Image.fromarray(grid).save("forward_diffusion.png")
    print("creating images...")
    timestep_images = [visualize_forward_diffusion(diffusors, scheduler_names, img, 3, i, True) for i in range(1000)]
    print("creating video...")
    create_video(timestep_images, 0, 120, "videos/forward_diffusion.mp4")
    