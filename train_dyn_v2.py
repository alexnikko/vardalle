from dataset_generator import generate_random_image
from utils import seed_everything
from config import generate_params

from tqdm.auto import tqdm

import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import ToPILImage

import numpy as np
from collections import defaultdict
import os
import shutil
import torchvision


torch.set_num_threads(4)

# CONSTS
seed = 42

# train params
device = 'cuda:2'
batch_size = 256
n_epochs = 1_000
epoch_len = 64
r = 3e-5

# model params
model_params = dict(
    input_channels=3,
    n_hid=64,
    n_downsamples=4,
    n_bottlenecks=4,
    codebook_size=32,
    code_size=64,
    num_codebooks=8,
)

# optimizer params
lr = 1e-4

# nc = num_codebooks, cs = codebook_size
# save params
saveroot = './snapshots'
savedir = f'train_orig_cs{model_params["codebook_size"]}_nc{model_params["num_codebooks"]}_16x16'
savename = 'snapshot.tar'
savepath = os.path.join(saveroot, savedir, savename)

samples_saveroot = f'./samples/{savedir}'




# samples_saveroot = f'samples/train_orig_cs{model_params["codebook_size"]}/'

class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, num_codebooks):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_codebooks = num_codebooks
        self.n_embed = n_embed

        # self.temperature = 1.0
        self.kld_scale = 5e-4

        self.proj = nn.Conv2d(num_hiddens, num_codebooks * n_embed, 1)
        self.embed = nn.Embedding(num_codebooks * n_embed, num_codebooks * embedding_dim)

    def forward(self, z, tau=1):

        # force hard = True when we are in eval mode, as we must quantize
        # hard = self.straight_through if self.training else True
        hard = True

        batch, _, height, width = z.shape

        logits = self.proj(z)
        logits = logits.view(batch, self.num_codebooks, self.n_embed, height, width)

        soft_one_hot = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-3)
        soft_one_hot = soft_one_hot.view(batch, -1, height, width)
        z_q = torch.einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=-3)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=-3).mean()

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

    def __init__(self, input_channels=3, n_hid=64, n_downsamples=3, n_bottlenecks=2,
                       codebook_size=16, code_size=32, num_codebooks=4):
        super().__init__()
        self.encoder = DeepMindEncoder(input_channels=input_channels, n_hid=n_hid,
                                       n_downsamples=n_downsamples, n_bottlenecks=n_bottlenecks)
        self.decoder = DeepMindDecoder(output_channels=input_channels, n_init=num_codebooks * code_size, n_hid=n_hid,
                                       n_upsamples=n_downsamples, n_bottlenecks=n_bottlenecks)
        self.quantizer = GumbelQuantize(self.encoder.output_channels, codebook_size, code_size, num_codebooks)


    def forward(self, x, tau=1):
        z = self.encoder(x)
        z_q, latent_loss = self.quantizer(z, tau=tau)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss


class CustomDataLoader():
    def __init__(self, batch_size, transform, inv_transform, generate_params):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.inv_transform = inv_transform
        self.generate_params = generate_params
        self.to_pil = ToPILImage()
    
    def get_batch(self, device):
        images = [self.transform(generate_random_image(**self.generate_params)) for _ in range(self.batch_size)]
        images = torch.stack(images)
        images = images.to(device)
        return images
    
    def recover_images(self, images):
        return self.inv_transform(images.detach().cpu())


def train(model, data: CustomDataLoader, optimizer, device, n_epochs, epoch_len, validation_data=None):
    metrics = defaultdict(list)
    metrics_names = ['recon_loss', 'latent_loss', 'loss']

    for epoch in range(n_epochs):
        for epoch_step in tqdm(range(epoch_len)):
            step = epoch * epoch_len + epoch_step
            tau = max(0.3, np.exp(-r * step))

            images = data.get_batch(device)

            images_recon, latent_loss = model(images, tau=tau)

            recon_loss = F.mse_loss(images, images_recon)
            loss = recon_loss + latent_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for key, value in zip(metrics_names, [recon_loss.item(), latent_loss.item(), loss.item()]):
                metrics[key].append(value)
        
        if validation_data is not None:
            with torch.inference_mode():
                validation_recon, _ = model(validation_data.to(device))
            validation_data_recovered = data.recover_images(validation_data)
            validation_recon_recovered = data.recover_images(validation_recon)
            grid = torch.cat([validation_data_recovered, validation_recon_recovered], dim=0)
            grid = torchvision.utils.make_grid(grid, nrow=grid.size(0) // 2)
            grid = data.to_pil(grid)
            grid.save(os.path.join(samples_saveroot, f'epoch_{str(epoch).zfill(4)}.png'))

        print(f'Epoch #{epoch}:', end='\t')
        for key, value in metrics.items():
            print(f'{key} = {np.mean(value[-epoch_len:]):.5f}\t', end='\t')
        print(f'cur step = {step}\tcur tau = {tau:.5f}', end='\t')
        print()
        print('Saving model...')
        save_snapshot(model, metrics, savepath)


def save_snapshot(model, metrics, savepath):
    snapshot = {
        'model': model.state_dict(),
        'metrics': metrics,
        'model_params': model_params 
    }
    torch.save(snapshot, savepath)
    print('Model has been saved...')


def main():
    seed_everything(seed)

    try:
        os.makedirs(os.path.join(saveroot, savedir))
        os.makedirs(samples_saveroot)
    except:
        ans = input(f'Do you want to remove {savedir} run? y/n: ')
        if ans != 'y':
            exit(0)
        shutil.rmtree(os.path.join(saveroot, savedir))
        os.makedirs(os.path.join(saveroot, savedir))
        shutil.rmtree(samples_saveroot)
        os.makedirs(samples_saveroot)

    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: 2 * x - 1)  # 2 * x - 1 == (x - 0.5) / 0.5
    ])
    inv_transform = T.Compose([
        T.Lambda(lambda x: (x + 1) / 2),
        # T.ToPILImage()
    ])

    data = CustomDataLoader(batch_size, transform, inv_transform, generate_params)
    n_val_images = 5
    validation_data = torch.stack([data.transform(generate_random_image(**generate_params)) for _ in range(n_val_images)])

    model = VQVAE(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    train(model, data, optimizer, device=device, n_epochs=n_epochs, epoch_len=epoch_len, validation_data=validation_data)


if __name__ == '__main__':
    main()
