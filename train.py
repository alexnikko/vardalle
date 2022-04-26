from tkinter import TOP
from unittest import loader
from dataset_generator import generate_random_image, showImagesHorizontally
from utils import seed_everything
from model import VQVAE2
from config import generate_params

from tqdm.auto import tqdm

import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from collections import defaultdict
import os
import shutil


saveroot = './snapshots'
savedir = 'overfit_1codebook'
savename = 'snapshot.tar'

os.makedirs(os.path.join(saveroot, savedir), exist_ok=True)
savepath = os.path.join(saveroot, savedir, savename)


seed = 42
seed_everything(seed)

def get_batch(batch_size, transform, **generate_params):
    return torch.stack([transform(generate_random_image(**generate_params))
                        for _ in range(batch_size)])



# gumbel softmax params
tau = 0.1
hard = True

# regularizers
target_density = 0.1  # kl reg will aim to leave this fraction of logits
beta_max = 0.1  # kl strength from beta vae
beta_warmup_epochs = 200
cv_reg = 0.1  # load balancing strength from shazeer
eps=1e-6

# training
batch_size = 32
# device = 'cpu'
device = 'cuda:0'

num_channels = 3
num_codebooks = 1
codebook_size = 256 * 64
kernel_size = 1

print(f'CODEBOOK_SIZE = {codebook_size}')

# model = VQVAE(num_channels=num_channels, num_codebooks=num_codebooks, codebook_size=codebook_size, kernel_size=kernel_size)
model = VQVAE2(num_channels=num_channels, h_dim=256, num_codebooks=num_codebooks, codebook_size=codebook_size, kernel_size=kernel_size)
opt = torch.optim.Adam(model.parameters(), 1e-4)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)


model = model.to(device)
# data = MNIST(root='.', train=True, download=True).data.float().view(-1, 1, 28, 28) / 255
# epoch_i = 0
transform = ToTensor()


def train():
    n_epochs = 1_000
    n_epoch_steps = 128
    batch_size = 16

    metrics = defaultdict(list)

    for epoch in range(n_epochs):
        for epoch_step in tqdm(range(n_epoch_steps)):
            images = get_batch(batch_size=batch_size, transform=transform, **generate_params)
            images = images.to(device)

            x_rec, codes, logp = model(images, hard=hard, tau=tau)

            neg_llh = F.mse_loss(x_rec, images)
            # neg_llh = torch.mean(torch.abs(x_rec - images))
            
            # KL to target density
            total_patches = codes.shape[0] * np.prod(codes.shape[-2:])
            log_denominator = np.log(total_patches * model.coding.num_codebooks)
            logp_zero = torch.logsumexp(logp[..., 0, :, :].flatten(), 0) - log_denominator
            logp_nonzero = torch.logsumexp(logp[..., 1:, :, :].flatten(), 0) - log_denominator
            kl = - ((1 - target_density) * logp_zero + target_density * logp_nonzero)
            
            # CV reg
            probs = logp[..., 1:, :, :].exp()
            load = probs.mean(dim=(0, -2, -1))  # [num_codebooks, codebook_size]
            mean = load.mean()
            variance = torch.mean((load - mean) ** 2)
            cv_squared = variance / (mean ** 2 + eps)
            
            beta = beta_max * min(1.0, epoch / beta_warmup_epochs)
            loss = neg_llh + beta * kl + cv_reg * cv_squared
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(f"Epoch #{epoch} LLH={-neg_llh.item():.5f}, KL={kl.item():.5f}, mean p(0)={logp_zero.exp().item():.5f}, cv^2={cv_squared.item():.5f}")
        print('Saving model...')
        for key, value in zip(
            ['LLH', 'KL', 'mean p(0)', 'cv^2'],
            [-neg_llh.item(), kl.item(), logp_zero.exp().item(), cv_squared.item()]
        ):
            metrics[key].append(value)
        snapshot = {
            'model': model.state_dict(),
            'metrics': metrics
        }
        torch.save(snapshot, savepath)
        print('Model has been saved...')


def overfit():
    n_epochs = 10_000
    # n_epoch_steps = 128
    batch_size = 1

    metrics = defaultdict(list)

    n_samples = 1
    data = [transform(generate_random_image(**generate_params)) for _ in range(n_samples)]

    if os.path.exists('samples/overfit'):
        shutil.rmtree('samples/overfit')
    os.makedirs('samples/overfit')

    for epoch in range(n_epochs):
        images = data[0: 0 + batch_size]
        images = torch.stack(images)
        images = images.to(device)

        x_rec, codes, logp = model(images, hard=hard, tau=tau)

        neg_llh = F.mse_loss(x_rec, images)
        
        # KL to target density
        total_patches = codes.shape[0] * np.prod(codes.shape[-2:])
        log_denominator = np.log(total_patches * model.coding.num_codebooks)
        logp_zero = torch.logsumexp(logp[..., 0, :, :].flatten(), 0) - log_denominator
        logp_nonzero = torch.logsumexp(logp[..., 1:, :, :].flatten(), 0) - log_denominator
        kl = - ((1 - target_density) * logp_zero + target_density * logp_nonzero)
        
        # CV reg
        probs = logp[..., 1:, :, :].exp()
        load = probs.mean(dim=(0, -2, -1))  # [num_codebooks, codebook_size]
        mean = load.mean()
        variance = torch.mean((load - mean) ** 2)
        cv_squared = variance / (mean ** 2 + eps)
        
        beta = beta_max * min(1.0, epoch / beta_warmup_epochs)
        #loss = neg_llh + beta * kl + cv_reg * cv_squared
        loss = neg_llh
        loss.backward()
        opt.step()
        opt.zero_grad()

        if epoch == 0:
            image = images[0].detach().cpu()
            image = ToPILImage()(image)
            image.save(f'samples/overfit/inp_img_{str(epoch).zfill(4)}.png')

        if epoch % 10 == 0:
            image_recon = x_rec[0].detach().cpu()
            image_recon = ToPILImage()(image_recon)
            image_recon.save(f'samples/overfit/rec_img_{str(epoch).zfill(4)}.png')
        
        if epoch == 500:
            print('setting new lr')
            for g in opt.param_groups:
                g['lr'] = 1e-5
                print(g['lr'])

        # scheduler.step()

        

        print(f"Epoch #{epoch} LLH={-neg_llh.item():.5f}, KL={kl.item():.5f}, mean p(0)={logp_zero.exp().item():.5f}, cv^2={cv_squared.item():.5f}")
        # print('Saving model...')
        # for key, value in zip(
        #     ['LLH', 'KL', 'mean p(0)', 'cv^2'],
        #     [-neg_llh.item(), kl.item(), logp_zero.exp().item(), cv_squared.item()]
        # ):
        #     metrics[key].append(value)
        # snapshot = {
        #     'model': model.state_dict(),
        #     'metrics': metrics
        # }
        # torch.save(snapshot, savepath)
        # print('Model has been saved...')


if __name__ == '__main__':
    overfit()
