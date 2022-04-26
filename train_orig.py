from tkinter import TOP
from unittest import loader
from dataset_generator import generate_random_image, showImagesHorizontally
from utils import seed_everything
# from model import VQVAE2
from config import generate_params

from tqdm.auto import tqdm

import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from collections import defaultdict
import os
import shutil
import torchvision

class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, straight_through=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z):

        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True

        logits = self.proj(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        z_q = torch.einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        return z_q, diff, ind


class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out


class DeepMindEncoder(nn.Module):

    def __init__(self, input_channels=3, n_hid=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
        )

        self.output_channels = 2 * n_hid
        self.output_stide = 4

    def forward(self, x):
        return self.net(x)


class DeepMindDecoder(nn.Module):

    def __init__(self, n_init=32, n_hid=64, output_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
            nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class LogitLaplace:
    """ the Logit Laplace distribution log likelihood from OpenAI's DALL-E paper """
    logit_laplace_eps = 0.1

    @classmethod
    def inmap(cls, x):
        # map [0,1] range to [eps, 1-eps]
        return (1 - 2 * cls.logit_laplace_eps) * x + cls.logit_laplace_eps

    @classmethod
    def unmap(cls, x):
        # inverse map, from [eps, 1-eps] to [0,1], with clamping
        return torch.clamp((x - cls.logit_laplace_eps) / (1 - 2 * cls.logit_laplace_eps), 0, 1)

    @classmethod
    def nll(cls, x, mu_logb):
        raise NotImplementedError # coming right up


class Normal:
    """
    simple normal distribution with fixed variance, as used by DeepMind in their VQVAE
    note that DeepMind's reconstruction loss (I think incorrectly?) misses a factor of 2,
    which I have added to the normalizer of the reconstruction loss in nll(), we'll report
    number that is half of what we expect in their jupyter notebook
    """
    data_variance = 0.06327039811675479 # cifar-10 data variance, from deepmind sonnet code

    @classmethod
    def inmap(cls, x):
        return x - 0.5 # map [0,1] range to [-0.5, 0.5]

    @classmethod
    def unmap(cls, x):
        return torch.clamp(x + 0.5, 0, 1)

    @classmethod
    def nll(cls, x, mu):
        return ((x - mu)**2).mean() / (2 * cls.data_variance) #+ math.log(math.sqrt(2 * math.pi * cls.data_variance))

class VQVAE(nn.Module):

    def __init__(self, input_channels=3):
        super().__init__()
        self.encoder = DeepMindEncoder(input_channels=input_channels)
        self.decoder = DeepMindDecoder(output_channels=input_channels)
        self.quantizer = GumbelQuantize(self.encoder.output_channels, 2 ** 14, 32)

        # the data reconstruction loss in the ELBO
        # ReconLoss = {
        #     'l2': Normal,
        #     'logit_laplace': LogitLaplace,
        #     # todo: add vqgan
        # }[args.loss_flavor]
        # self.recon_loss = ReconLoss
        self.recon_loss = Normal


    def forward(self, x):
        z = self.encoder(x)
        z_q, latent_loss, ind = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss, ind


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
# tau = 0.1
# hard = True

# regularizers
# target_density = 0.1  # kl reg will aim to leave this fraction of logits
# beta_max = 0.1  # kl strength from beta vae
# beta_warmup_epochs = 200
# cv_reg = 0.1  # load balancing strength from shazeer
# eps=1e-6

# training
# batch_size = 32
# device = 'cpu'
device = 'cuda:0'

# num_channels = 3
# num_codebooks = 1
# codebook_size = 256 * 64
# kernel_size = 1

# print(f'CODEBOOK_SIZE = {codebook_size}')

# model = VQVAE(num_channels=num_channels, num_codebooks=num_codebooks, codebook_size=codebook_size, kernel_size=kernel_size)
# model = VQVAE2(num_channels=num_channels, h_dim=256, num_codebooks=num_codebooks, codebook_size=codebook_size, kernel_size=kernel_size)
model = VQVAE()
opt = torch.optim.Adam(model.parameters(), 1e-4)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)


model = model.to(device)
# data = MNIST(root='.', train=True, download=True).data.float().view(-1, 1, 28, 28) / 255
# epoch_i = 0
transform = ToTensor()


def train():
    n_epochs = 1_000
    n_epoch_steps = 128
    batch_size = 4

    metrics = defaultdict(list)
    if os.path.exists('samples/train_orig'):
        shutil.rmtree('samples/train_orig')
    os.makedirs('samples/train_orig')

    for epoch in range(n_epochs):
        for epoch_step in tqdm(range(n_epoch_steps)):
            images = get_batch(batch_size=batch_size, transform=transform, **generate_params)
            images = images.to(device)

            images = model.recon_loss.inmap(images)
            x_hat, latent_loss, ind = model(images)
            recon_loss = model.recon_loss.nll(images, x_hat)
            loss = recon_loss + latent_loss

            # neg_llh = F.mse_loss(x_rec, images)/
            # neg_llh = torch.mean(torch.abs(x_rec - images))
            
            # KL to target density
            # total_patches = codes.shape[0] * np.prod(codes.shape[-2:])
            # log_denominator = np.log(total_patches * model.coding.num_codebooks)
            # logp_zero = torch.logsumexp(logp[..., 0, :, :].flatten(), 0) - log_denominator
            # logp_nonzero = torch.logsumexp(logp[..., 1:, :, :].flatten(), 0) - log_denominator
            # kl = - ((1 - target_density) * logp_zero + target_density * logp_nonzero)
            
            # CV reg
            # probs = logp[..., 1:, :, :].exp()
            # load = probs.mean(dim=(0, -2, -1))  # [num_codebooks, codebook_size]
            # mean = load.mean()
            # variance = torch.mean((load - mean) ** 2)
            # cv_squared = variance / (mean ** 2 + eps)
            
            # beta = beta_max * min(1.0, epoch / beta_warmup_epochs)
            # loss = neg_llh + beta * kl + cv_reg * cv_squared
            loss.backward()
            opt.step()
            opt.zero_grad()
        images = model.recon_loss.unmap(images).detach().cpu()
        recon_images = model.recon_loss.unmap(x_hat).detach().cpu()
        print(images.shape, recon_images.shape)
        grid = torch.cat([images, recon_images], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=batch_size)
        grid = ToPILImage()(grid)
        grid.save(f'samples/train_orig/epoch_{str(epoch).zfill(4)}.png')
        # print(f"Epoch #{epoch} LLH={-neg_llh.item():.5f}, KL={kl.item():.5f}, mean p(0)={logp_zero.exp().item():.5f}, cv^2={cv_squared.item():.5f}")
        print(f"Epoch #{epoch} LLH={loss.item():.5f}")
        print('Saving model...')
        for key, value in zip(
            ['LLH'],
            [loss.item()]
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
        images = model.recon_loss.inmap(images)
        x_hat, latent_loss, ind = model(images)
        recon_loss = model.recon_loss.nll(images, x_hat)
        loss = recon_loss + latent_loss



        # x_rec, codes, logp = model(images, hard=hard, tau=tau)

        # neg_llh = F.mse_loss(x_rec, images)
        
        # # KL to target density
        # total_patches = codes.shape[0] * np.prod(codes.shape[-2:])
        # log_denominator = np.log(total_patches * model.coding.num_codebooks)
        # logp_zero = torch.logsumexp(logp[..., 0, :, :].flatten(), 0) - log_denominator
        # logp_nonzero = torch.logsumexp(logp[..., 1:, :, :].flatten(), 0) - log_denominator
        # kl = - ((1 - target_density) * logp_zero + target_density * logp_nonzero)
        
        # # CV reg
        # probs = logp[..., 1:, :, :].exp()
        # load = probs.mean(dim=(0, -2, -1))  # [num_codebooks, codebook_size]
        # mean = load.mean()
        # variance = torch.mean((load - mean) ** 2)
        # cv_squared = variance / (mean ** 2 + eps)
        
        # beta = beta_max * min(1.0, epoch / beta_warmup_epochs)
        #loss = neg_llh + beta * kl + cv_reg * cv_squared
        # loss = neg_llh
        loss.backward()
        opt.step()
        opt.zero_grad()

        if epoch == 0:
            image = images[0].detach().cpu()
            image = ToPILImage()(image)
            image.save(f'samples/overfit/inp_img_{str(epoch).zfill(4)}.png')

        if epoch % 10 == 0:
            image_recon = model.recon_loss.unmap(x_hat[0]).detach().cpu()
            image_recon = ToPILImage()(image_recon)
            image_recon.save(f'samples/overfit/rec_img_{str(epoch).zfill(4)}.png')
        
        if epoch == 500:
            print('setting new lr')
            for g in opt.param_groups:
                g['lr'] = 1e-5
                print(g['lr'])

        # scheduler.step()

        

        # print(f"Epoch #{epoch} LLH={-neg_llh.item():.5f}, KL={kl.item():.5f}, mean p(0)={logp_zero.exp().item():.5f}, cv^2={cv_squared.item():.5f}")
        print(f"Epoch #{epoch} LLH={loss.item():.5f}")
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
    # overfit()
    train()
