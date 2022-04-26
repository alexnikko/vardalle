from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class GumbelCoding2d(nn.Module):
    def __init__(self, in_features: int, num_codebooks: int, codebook_size: int, **kwargs):
        super().__init__()
        self.num_codebooks, self.codebook_size = num_codebooks, codebook_size
        self.proj = nn.Conv2d(in_features, num_codebooks * codebook_size, **kwargs)
        self.embeddings = nn.Embedding(codebook_size, in_features)
        
    def forward(self, x, **kwargs):
        batch, _, height, width = x.shape
        logits = self.proj(x).view(batch, self.num_codebooks, self.codebook_size, height, width)
        codes = F.gumbel_softmax(logits, dim=-3, **kwargs)  # gumbel over codebook size
        # print(codes.shape)
        # print(codes.sum())
        # print(codes[0, 0].argmax(0))
        # print(codes[0, 0].argmax(0).shape)
        # assert False
        embs = codes.permute(0, 1, 3, 4, 2)
        embs = embs.reshape(-1, self.codebook_size)
        embs = embs.argmax(-1)
        # print(embs.shape)
        embs = self.embeddings(embs)
        # print(embs.shape)
        embs = embs.reshape(batch, height, width, -1)
        embs = embs.permute(0, 3, 1, 2)
        # print(embs.shape)
        # assert False
        return codes, F.log_softmax(logits, dim=-3), embs

# actually it is not
class VQVAE(nn.Module):
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
        
        decoder_input_channels = 512
        # decoder_input_channels = self.coding.num_codebooks * self.coding.codebook_size
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_input_channels, 512, 3, padding=2), nn.BatchNorm2d(512), nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(512, 256, 3, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(256, 128, 3, padding=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(128, 64, 3, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, num_channels, 1)
        )
        
    def forward(self, x, **kwargs):
        codes, logp, embs = self.coding(self.encoder(x), **kwargs)
        return self.decoder(embs), codes, logp
        # return self.decoder(codes.flatten(1, 2)), codes, logp


class ResBlock(nn.Module):
    def __init__(self, in_dim, exp_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, exp_dim, 3, padding=1),
            nn.BatchNorm2d(exp_dim),
            nn.ReLU(),
            nn.Conv2d(exp_dim, in_dim, 1)
        )
    
    def forward(self, x):
        return F.relu(self.conv(x) + x)


class Encoder(nn.Module):
    def __init__(self, input_channels, h_dim, n_downsamples=3, n_bottlenecks=2):
        super().__init__()
        net = []
        for i in range(n_downsamples):
            net.extend([
                nn.Conv2d(input_channels, h_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.MaxPool2d(2),
                nn.ReLU()
            ])
            input_channels = h_dim
            h_dim *= 2
        h_dim = input_channels
        for i in range(n_bottlenecks):
            net.append(ResBlock(h_dim, h_dim // 2))
        self.net = nn.Sequential(*net)
        

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, output_channels, h_dim, n_upsamples=3):
        super().__init__()
        net = []
        h_dim = h_dim * 2 ** (n_upsamples - 1)
        net.append(ResBlock(h_dim, h_dim // 2))
        for i in range(n_upsamples):
            net.append(nn.UpsamplingNearest2d(scale_factor=2))
            net.append(nn.Conv2d(h_dim, h_dim // 2, kernel_size=3, padding=1))
            net.append(nn.ReLU())
            h_dim //= 2
        net.append(ResBlock(h_dim, h_dim))
        net.append(nn.Conv2d(h_dim, output_channels, kernel_size=1))
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x)


class VQVAE2(nn.Module):
    def __init__(self, num_channels, h_dim, **kwargs):
        super().__init__()

        self.encoder = Encoder(input_channels=num_channels, h_dim=h_dim,
                               n_downsamples=3, n_bottlenecks=2)
        # print(self.encoder)
        self.coding = GumbelCoding2d(in_features=h_dim * 2 ** (3 - 1), **kwargs)
        # print(self.coding)

        self.decoder = Decoder(output_channels=num_channels, h_dim=h_dim, n_upsamples=3)
    
    def forward(self, x, **kwargs):
        codes, logp, embs = self.coding(self.encoder(x), **kwargs)
        return self.decoder(embs), codes, logp
         


if __name__ == '__main__':
    x = torch.rand(10, 3, 256, 256)
    enc = Encoder(3, h_dim=256)
    dec = Decoder(3, h_dim=256)
    z = enc(x)
    x_rec = dec(z)
    print(x.shape, z.shape, x_rec.shape, sep='\n')
