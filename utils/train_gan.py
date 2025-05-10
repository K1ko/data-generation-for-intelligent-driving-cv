import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils as vutils
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import os
from your_data_loader import SignsDataLoader  # Replace with your actual dataset class
import numpy as np

# Generator Network
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Training Function
def train_gan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = SignsDataLoader(args.dataset_path, transform=data_transforms)
    filtered_dataset = [sample for sample in dataset if sample[1] == args.target_class]
    train_loader = DataLoader(filtered_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Models
    generator = Generator(args.nz, args.ngf).to(device)
    discriminator = Discriminator(args.ndf).to(device)

    # Optimizers and loss
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    for epoch in range(args.num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            b_size = real_images.size(0)
            real_labels = torch.ones(b_size, 1, device=device)
            fake_labels = torch.zeros(b_size, 1, device=device)

            # Train Discriminator
            discriminator.zero_grad()
            output_real = discriminator(real_images).view(-1, 1)
            loss_real = criterion(output_real, real_labels)
            loss_real.backward()

            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach()).view(-1, 1)
            loss_fake = criterion(output_fake, fake_labels)
            loss_fake.backward()
            optimizerD.step()

            # Train Generator
            generator.zero_grad()
            output = discriminator(fake_images).view(-1, 1)
            lossG = criterion(output, real_labels)
            lossG.backward()
            optimizerG.step()

            if i % 5 == 0:
                print(f"[{epoch}/{args.num_epochs}] [{i}/{len(train_loader)}] "
                      f"Loss_D: {(loss_real + loss_fake).item():.4f}, Loss_G: {lossG.item():.4f}")

        if epoch in [0, args.num_epochs // 4, args.num_epochs - 1]:
            with torch.no_grad():
                fake_images = generator(fixed_noise).cpu()
                grid = vutils.make_grid(fake_images, normalize=True)
                plt.imshow(grid.permute(1, 2, 0))
                plt.title(f"Epoch {epoch + 1}")
                plt.axis('off')
                plt.show()

        if epoch == args.num_epochs - 1:
            output_dir = Path(args.output_dir) / str(args.target_class)
            output_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                for idx, image in enumerate(generator(fixed_noise).cpu()):
                    img = (image + 1) / 2
                    F.to_pil_image(img).save(output_dir / f"synth_GAN_{idx+1:04d}_{args.target_class}.png")
            print(f"Saved {len(fake_images)} images to {output_dir}")

    print("Training complete.")

# argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the GTSRB training data")
    parser.add_argument("--output_dir", type=str, default="./synth_output", help="Directory to save generated images")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--target_class", type=int, default=5)

    args = parser.parse_args()
    train_gan(args)
