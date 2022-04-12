from dataset_generator import generate_random_image, showImagesHorizontally, generate_params
import random

from tqdm.auto import tqdm

import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
from collections import defaultdict


seed = 42
np.random.seed(seed)
random.seed(seed)



class GumbelCoding2d(nn.Module):
    def __init__(self, in_features: int, num_codebooks: int, codebook_size: int, **kwargs):
        super().__init__()
        self.num_codebooks, self.codebook_size = num_codebooks, codebook_size
        self.proj = nn.Conv2d(in_features, num_codebooks * codebook_size, **kwargs)
        
    def forward(self, x, **kwargs):
        batch, _, height, width = x.shape
        logits = self.proj(x).view(batch, self.num_codebooks, self.codebook_size, height, width)
        codes = F.gumbel_softmax(logits, dim=-3, **kwargs)  # gumbel over codebook size
        return codes, F.log_softmax(logits, dim=-3)

    
# def iterate_minibatches(*tensors, batch_size, shuffle=True, epochs=1,
#                         allow_incomplete=True, callback=lambda x:x):
#     indices = np.arange(len(tensors[0]))
#     upper_bound = int((np.ceil if allow_incomplete else np.floor) (len(indices) / batch_size)) * batch_size
#     epoch = 0
#     while True:
#         if shuffle:
#             np.random.shuffle(indices)
#         for batch_start in callback(range(0, upper_bound, batch_size)):
#             batch_ix = indices[batch_start: batch_start + batch_size]
#             batch = [tensor[batch_ix] for tensor in tensors]
#             yield batch if len(tensors) > 1 else batch[0]
#         epoch += 1
#         if epoch >= epochs:
#             break


class Foo(nn.Module):
    def __init__(self, num_channels: int, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.ReLU(),
            nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.ReLU(),
            nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.MaxPool2d(2), nn.ReLU(),
            nn.Conv2d(256, 512, 3), nn.BatchNorm2d(512), nn.ReLU(),
        )
        
        self.coding = GumbelCoding2d(in_features=512, **kwargs)
        
        decoder_input_channels = self.coding.num_codebooks * self.coding.codebook_size
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_input_channels, 512, 3, padding=2), nn.BatchNorm2d(512), nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(512, 256, 3, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(256, 128, 3, padding=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(128, 64, 3, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, num_channels, 1)
        )
        
    def forward(self, x, **kwargs):
        codes, logp = self.coding(self.encoder(x), **kwargs)
        return self.decoder(codes.flatten(1, 2)), codes, logp



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
num_codebooks = 64
codebook_size = 128
kernel_size = 1

model = Foo(num_channels=num_channels, num_codebooks=num_codebooks, codebook_size=codebook_size, kernel_size=kernel_size)
opt = torch.optim.Adam(model.parameters(), 1e-4)


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
        torch.save(snapshot, 'snapshot.tar')
        # torch.save(model.state_dict(), 'checkpoint.pth')
        print('Model has been saved...')


if __name__ == '__main__':
    train()
    # snapshot = torch.load('snapshot.tar', map_location='cpu')
    # for key, value in snapshot['metrics'].items():
        # print(f'{key}: {value}')
    
    # model.cpu().load_state_dict(snapshot['model'])
    # print('OK')
    # print(snapshot['model'])
    # train()
