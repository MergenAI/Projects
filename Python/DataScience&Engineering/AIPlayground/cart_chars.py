import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Dataset Transformation
transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),  # Random rotation
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

# Load the dataset
data_dir = "data/cartoonset100k_jpg"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Use DataLoader for batch processing and shuffling
batch_size = 128
latent_space = 256  # Increased latent space dimension for richer representation
epochs = 15  # Increase training epochs to allow better learning
lr = 1e-3  # Lower learning rate for more stable training

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
print(f"Total number of images: {len(dataset)}")

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)  # Output: (128, 32, 32)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)  # Output: (64, 16, 16)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # Output: (64, 8, 8)
        self.conv6 = nn.Conv2d(64, 8, kernel_size=3, stride=2, padding=1)  # Output: (8, 4, 4)
        self.flatten = nn.Flatten()
        dim = 8 * 4 * 4
        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.mu = nn.Linear(64, latent_space)
        self.logvar = nn.Linear(64, latent_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv6(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar

# Reparametrization Trick
def reparametrize(mu, logvar):
    eps = torch.randn_like(logvar)
    std = torch.exp(0.5 * logvar)
    return mu + std * eps

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        dim = 8 * 4 * 4
        self.fc1 = nn.Linear(latent_space, dim)
        self.fc2 = nn.Linear(dim, 16 * 4 * 4)
        self.convT1 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: (32, 8, 8)
        self.convT2 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: (64, 16, 16)
        self.convT3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: (64, 32, 32)
        self.convT4 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: (128, 64, 64)
        self.convT5 = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1)  # Output: (3, 64, 64)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 16, 4, 4)  # Reshape to match the ConvTranspose2d input shape
        x = F.relu(self.convT1(x))
        x = F.relu(self.convT2(x))
        x = F.relu(self.convT3(x))
        x = F.relu(self.convT4(x))
        x = torch.tanh(self.convT5(x))  # Use tanh for final activation to keep values between [-1, 1]
        return x

# Loss Function with β-VAE for Better KL Loss Control
def loss_function(x, recon_x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    kl_loss *= beta
    return recon_loss + kl_loss

# Model Class
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparametrize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# Training Loop with KL Loss Warm-Up
beta_start = 0.01
beta_end = 1.0
warmup_epochs = 50
# Main Training Block
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        beta = min(beta_end, beta_start + (beta_end - beta_start) * (epoch / warmup_epochs))
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            batch_loss = loss_function(data, recon, mu, logvar, beta)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader.dataset):.4f}, Beta: {beta:.4f}")

    # Save the VAE model
    torch.save(model.state_dict(), 'models/vae.pth')

    # Load model and generate new images
    model.load_state_dict(torch.load('models/vae.pth', map_location=device))
    model.eval()

    num_samples = 10  # Number of new instances to generate
    with torch.no_grad():
        # Sample from the latent space (standard normal distribution)
        z = torch.randn(num_samples, latent_space).to(device)

        # Use the decoder to generate images
        generated_images = model.decoder(z)

        # Plot generated images
        for i in range(num_samples):
            img = generated_images[i].cpu().numpy()
            img = (img * 0.5) + 0.5  # Denormalize to [0, 1]
            img = np.transpose(img, (1, 2, 0))  # Change shape from (C, H, W) to (H, W, C)
            plt.imshow(img)
            plt.title(f"Generated Image {i + 1}")
            plt.axis('off')
            plt.show()
