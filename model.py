"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(
        beta_start,
        beta_end,
        T,
        dtype=torch.float32
    )
    # TODO: return a linear beta schedule of length T
    pass

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1.0- betas
    pass

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas,dim=0)
    pass

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    B = t.shape[0]
    out = a[t]
    return out.reshape(B, 1, 1, 1)

    pass

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    alpha_bar_t = extract_into_batch(alphas_cumprod, t, x0)

    return (
        torch.sqrt(alpha_bar_t) * x0
        + torch.sqrt(1.0 - alpha_bar_t) * noise
    )

    pass

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    betas = linear_beta_schedule(T, beta_start, beta_end)
    
    alphas = alphas_from_betas(betas)
    
    alphas_cumprod = cumprod_alphas(alphas)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "T": T,
    }
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts

    pass

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    return ((noise - noise_pred) ** 2).mean()
    # TODO: MSE between predicted and true noise
    pass

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    xt = q_sample(x0, t, noise, alphas_cumprod)

    # Predict the noise
    noise_pred = model(xt, t)

    # Calculate MSE loss
    return noise_prediction_loss(noise_pred, noise)
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    pass

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    half = dim // 2

    if half == 1:
        exponent = torch.zeros(1, device=t.device)
    else:
        exponent = torch.arange(
            half, device=t.device, dtype=torch.float32
        ) / (half - 1)

    frequencies = 1.0 / (10000 ** exponent)

    angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)

    emb = torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=1
    )

    return emb
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    pass

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(
    in_ch: int = 1,
    hidden: int = 16,
    time_dim: int = 16,
    seed: int = 0
) -> dict:
    
    torch.manual_seed(seed)

    def weight(shape):
        return (torch.randn(*shape) * 0.02).requires_grad_()

    def bias(shape):
        return torch.zeros(*shape, requires_grad=True)

    return {
        "conv_in_w": weight((hidden, in_ch, 3, 3)),
        "conv_in_b": bias((hidden,)),

        "time_mlp_w": weight((hidden, time_dim)),
        "time_mlp_b": bias((hidden,)),

        "conv_mid_w": weight((hidden, hidden, 3, 3)),
        "conv_mid_b": bias((hidden,)),

        "conv_out_w": weight((in_ch, hidden, 3, 3)),
        "conv_out_b": bias((in_ch,)),
    }

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # 1. Input convolution
    h = F.conv2d(
        x,
        params["conv_in_w"],
        params["conv_in_b"],
        padding=1
    )

    # 2. Timestep embedding + time MLP
    temb = timestep_embedding(t, params["time_mlp_w"].shape[1])

    temb = F.relu(
        F.linear(
            temb,
            params["time_mlp_w"],
            params["time_mlp_b"]
        )
    )

    # Add time embedding to every spatial position
    h = h + temb[:, :, None, None]

    # 3. Middle convolution + ReLU
    h = F.relu(
        F.conv2d(
            h,
            params["conv_mid_w"],
            params["conv_mid_b"],
            padding=1
        )
    )

    # 4. Output convolution
    return F.conv2d(
        h,
        params["conv_out_w"],
        params["conv_out_b"],
        padding=1
    )
    # TODO: time-conditioned tiny CNN predicting noise
    pass

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    torch.manual_seed(seed)

    images = torch.zeros(n, 1, size, size)
    radius = size // 4

    yy, xx = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing="ij"
    )

    for i in range(n):
        center = torch.randint(radius, size - radius, (2,))
        cy, cx = center[0], center[1]

        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        images[i, 0][mask] = 1.0

    return images
    # TODO: n images with a random bright disk on a black background
    pass

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(
    params: dict,
    x0,
    schedule: dict,
    lr: float = 1e-2,
    seed: int = 0
):
    # Seed RNG for deterministic timestep and noise sampling
    torch.manual_seed(seed)

    B = x0.shape[0]
    T = schedule["T"]

    # Sample timesteps uniformly from [0, T)
    t = torch.randint(0, T, (B,), device=x0.device)

    # Sample Gaussian noise
    noise = torch.randn_like(x0)

    # Compute DDPM noise-prediction loss
    loss = diffusion_training_loss(
        lambda x, t: tiny_unet_forward(x, t, params),
        x0,
        t,
        noise,
        schedule["alphas_cumprod"],
    )

    # Backpropagation
    loss.backward()

    # SGD update, creating fresh leaf tensors
    new_params = {}

    for name, p in params.items():
        if p.grad is not None:
            new_params[name] = (
                p - lr * p.grad
            ).detach().requires_grad_(True)
        else:
            new_params[name] = p.detach().clone().requires_grad_(True)

    return new_params, float(loss)    # TODO: sample t,noise -> loss -> SGD on params
    pass

# Step 14 - train_ddpm
import torch
import torch.nn.functional as F

def train_ddpm(
    dataset,
    params: dict,
    schedule: dict,
    num_steps: int = 50,
    batch_size: int = 16,
    lr: float = 1e-2,
    seed: int = 0
):
    history = []

    for step in range(num_steps):
        # Deterministic minibatch sampling
        torch.manual_seed(seed + step)

        indices = torch.randint(
            0,
            len(dataset),
            (batch_size,)
        )

        x0 = dataset[indices]

        # One DDPM training step
        params, loss = ddpm_train_step(
            params,
            x0,
            schedule,
            lr=lr,
            seed=seed + step
        )

        history.append(float(loss))

    return params, history    # TODO: minibatch SGD training loop
    pass

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    alpha_bar_t = extract_into_batch(alphas_cumprod, t, x_t)

    return (
        x_t - torch.sqrt(1.0 - alpha_bar_t) * eps
    ) / torch.sqrt(alpha_bar_t)    # TODO: invert the q_sample equation for x0
    pass

# Step 16 - ddpm_p_mean_variance
import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    # Predict clean image
    x0_hat = predict_x0_from_eps(
        x_t, t, eps, schedule["alphas_cumprod"]
    ).clamp(-1, 1)

    # Current timestep values
    alpha_t = extract_into_batch(schedule["alphas"], t, x_t)
    alpha_bar_t = extract_into_batch(
        schedule["alphas_cumprod"], t, x_t
    )
    beta_t = extract_into_batch(schedule["betas"], t, x_t)

    # alpha_bar_(t-1), with alpha_bar_-1 = 1
    alphas_cumprod_prev = torch.cat([
        torch.ones(
            1,
            dtype=schedule["alphas_cumprod"].dtype,
            device=schedule["alphas_cumprod"].device
        ),
        schedule["alphas_cumprod"][:-1]
    ])

    alpha_bar_prev = extract_into_batch(
        alphas_cumprod_prev, t, x_t
    )

    # Posterior mean
    mean = (
        torch.sqrt(alpha_bar_prev) * beta_t
        / (1.0 - alpha_bar_t)
    ) * x0_hat + (
        torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)
        / (1.0 - alpha_bar_t)
    ) * x_t

    # Simplified fixed variance
    variance = beta_t

    return mean, variance, x0_hat    # TODO: return (posterior_mean, variance, x0_hat)
    pass

# Step 17 - ddpm_p_sample
import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    # 1. Predict noise
    eps = tiny_unet_forward(x_t, t, params)

    # 2. Get reverse-process mean and variance
    mean, var, _ = ddpm_p_mean_variance(
        x_t, t, eps, schedule
    )

    # Generate noise if none was provided
    if noise is None:
        noise = torch.randn_like(x_t)

    # No noise should be added when t == 0
    nonzero_mask = (t != 0).float().reshape(
        t.shape[0], *([1] * (x_t.ndim - 1))
    )

    # 3. Sample x_(t-1)
    x_prev = mean + nonzero_mask * torch.sqrt(var) * noise

    return x_prev    # TODO: one reverse step x_t -> x_{t-1}
    pass

# Step 18 - ddpm_sample_loop
import torch
import torch.nn.functional as F

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    torch.manual_seed(seed)

    # Start from pure Gaussian noise
    x = torch.randn(shape)

    T = schedule["T"]

    # Reverse diffusion: T-1 -> 0
    for step in reversed(range(T)):
        t_batch = torch.full(
            (shape[0],),
            step,
            dtype=torch.long
        )

        x = ddpm_p_sample(
            x,
            t_batch,
            params,
            schedule
        )

    return x    # TODO: ancestral sampling from pure noise to x0
    pass

# Step 19 - sample_quality_mse
import torch
import torch.nn.functional as F

def sample_quality_mse(samples, dataset) -> float:
    # Flatten each image
    samples_flat = samples.flatten(start_dim=1)
    dataset_flat = dataset.flatten(start_dim=1)

    # Pairwise squared differences: (N, M, features)
    diff = samples_flat[:, None, :] - dataset_flat[None, :, :]

    # MSE for every sample-dataset pair: (N, M)
    mse = (diff ** 2).mean(dim=-1)

    # Nearest dataset image for each generated sample
    min_mse = mse.min(dim=1).values

    # Mean of per-sample minimum MSEs
    return float(min_mse.mean())    # TODO: mean over samples of min MSE to any dataset image
    pass

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

