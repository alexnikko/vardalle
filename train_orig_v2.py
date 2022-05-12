from tkinter import TOP
from unittest import loader

from sklearn import naive_bayes
from dataset_generator import generate_random_image, showImagesHorizontally
from utils import seed_everything
# from model import VQVAE2
from config import generate_params

from tqdm.auto import tqdm

import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from collections import defaultdict
import os
import shutil
import torchvision

codebook_size = 32  #1#2##8#16 # 2 ** 14 # 16k


# CONSTS
seed = 42
device = 'cuda:0'

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

        # ind = soft_one_hot.argmax(dim=1)
        return z_q, diff


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

    def __init__(self, input_channels=3, n_hid=64, n_downsamples=3, n_bottlenecks=2):
        super().__init__()
        self.input_channels = input_channels
        self.n_hid = n_hid
        self.n_downsamples = n_downsamples
        self.n_bottlenecks = n_bottlenecks

        downsample_layers = []
        in_channels = input_channels
        out_channels = n_hid
        for i in range(n_downsamples):
            downsample_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
            out_channels = 2 * out_channels
        before_bottleneck = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ]
        bottleneck = [ResBlock(in_channels, in_channels // 4) for _ in range(n_bottlenecks)]

        self.net = nn.Sequential(*(downsample_layers + before_bottleneck + bottleneck))
        self.output_channels = in_channels
        self.output_stide = 4

    def forward(self, x):
        return self.net(x)


class DeepMindDecoder(nn.Module):

    def __init__(self, n_init=32, n_hid=64, output_channels=3, n_upsamples=3, n_bottlenecks=2):
        super().__init__()
        self.n_init = n_init
        self.n_hid = n_hid
        self.output_channels = output_channels
        self.n_upsamples = n_upsamples
        self.n_bottlenecks = n_bottlenecks

        out_channels = n_hid * 2 ** (n_upsamples - 1)
        first = [
            nn.Conv2d(n_init, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ]
        in_channels = out_channels
        out_channels = out_channels // 2
        bottlenecks = [ResBlock(in_channels, in_channels // 4) for _ in range(n_bottlenecks)]
        upsample_layers = []
        for i in range(n_upsamples):
            upsample_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            if i == n_upsamples - 1:
                upsample_layers.append(nn.Tanh())
            else:
                
                upsample_layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            out_channels = out_channels // 2
            if i == n_upsamples - 2:
                out_channels = output_channels
        
        self.net = nn.Sequential(*(first + bottlenecks + upsample_layers))

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):

    def __init__(self, input_channels=3, n_hid=64, n_downsamples=3, n_bottlenecks=2, codebook_size=16, code_size=32):
        super().__init__()
        self.encoder = DeepMindEncoder(input_channels=input_channels, n_hid=n_hid,
                                       n_downsamples=n_downsamples, n_bottlenecks=n_bottlenecks)
        self.decoder = DeepMindDecoder(output_channels=input_channels, n_init=code_size, n_hid=n_hid,
                                       n_upsamples=n_downsamples, n_bottlenecks=n_bottlenecks)
        self.quantizer = GumbelQuantize(self.encoder.output_channels, codebook_size, code_size)

        # the data reconstruction loss in the ELBO
        # ReconLoss = {
        #     'l2': Normal,
        #     'logit_laplace': LogitLaplace,
        #     # todo: add vqgan
        # }[args.loss_flavor]
        # self.recon_loss = ReconLoss
        # self.recon_loss = Normal


    def forward(self, x):
        z = self.encoder(x)
        z_q, latent_loss = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss


saveroot = './snapshots'
savedir = 'train_orig_1cs'
savename = 'snapshot.tar'

os.makedirs(os.path.join(saveroot, savedir), exist_ok=True)
savepath = os.path.join(saveroot, savedir, savename)


class CustomDataLoader():
    def __init__(self, batch_size, transform, inv_transform, epoch_len, generate_params):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.inv_transform = inv_transform
        self.generate_params = generate_params
    
    def get_batch(self, device):
        images = [self.transform(generate_random_image(**self.generate_params) for _ in range(self.batch_size))]
        images = torch.stack(images)
        return images
    
    def recover_images(self, images):
        return [self.inv_transform(image) for image in images]



# def get_batch(batch_size, transform, **generate_params):
#     return torch.stack([transform(generate_random_image(**generate_params))
#                         for _ in range(batch_size)])



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


def train(model, optimizer):
    n_epochs = 1_000
    n_epoch_steps = 128
    batch_size = 4

    metrics = defaultdict(list)
    if os.path.exists('samples/train_orig_1cs'):
        shutil.rmtree('samples/train_orig_1cs')
    os.makedirs('samples/train_orig_1cs')

    for epoch in range(n_epochs):
        for epoch_step in tqdm(range(n_epoch_steps)):
            images = get_batch(batch_size=batch_size, transform=transform, **generate_params)
            images = images.to(device)

            images = model.recon_loss.inmap(images)
            x_hat, latent_loss, ind = model(images)
            recon_loss = model.recon_loss.nll(images, x_hat)
            loss = recon_loss + latent_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        images = model.recon_loss.unmap(images).detach().cpu()
        recon_images = model.recon_loss.unmap(x_hat).detach().cpu()
        grid = torch.cat([images, recon_images], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=batch_size)
        grid = ToPILImage()(grid)
        grid.save(f'samples/train_orig_1cs/epoch_{str(epoch).zfill(4)}.png')
        # print(f"Epoch #{epoch} LLH={-neg_llh.item():.5f}, KL={kl.item():.5f}, mean p(0)={logp_zero.exp().item():.5f}, cv^2={cv_squared.item():.5f}")
        print(f"Epoch #{epoch} LLH={loss.item():.5f}")
        print('Saving model...')
        for key, value in zip(
            ['LLH'],
            [loss.item()]
        ):
            metrics[key].append(value)
        save_snapshot(model, metrics, savepath)


def save_snapshot(model, metrics, savepath):
    snapshot = {
        'model': model.state_dict(),
        'metrics': metrics
    }
    torch.save(snapshot, savepath)
    print('Model has been saved...')


def main():
    seed_everything(seed)
    model = VQVAE()
    opt = torch.optim.Adam(model.parameters(), 1e-4)
    model = model.to(device)
    
    transform = ToTensor()


if __name__ == '__main__':
    train()
