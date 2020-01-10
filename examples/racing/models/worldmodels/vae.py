
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size, m, shrink=1):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, m)
        self.deconv1 = nn.ConvTranspose2d(m, int(128 * shrink), 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(int(128 * shrink), int(64 * shrink), 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(int(64 * shrink), int(32 * shrink), 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(int(32 * shrink), img_channels, 6, stride=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction


class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, img_channels, latent_size, m, shrink=1):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, int(32 * shrink), 4, stride=2)
        self.conv2 = nn.Conv2d(int(32 * shrink), int(64 * shrink), 4, stride=2)
        self.conv3 = nn.Conv2d(int(64 * shrink), int(128 * shrink), 4, stride=2)
        self.conv4 = nn.Conv2d(int(128 * shrink), int(256 * shrink), 4, stride=2)

        self.fc_mu = nn.Linear(m, latent_size)
        self.fc_logsigma = nn.Linear(m, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma


class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size, m, shrink=1):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size, m, shrink)
        self.decoder = Decoder(img_channels, latent_size, m, shrink)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
