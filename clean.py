import argparse
import os
import pathlib
import re

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib.pyplot as plt
from dataset import MNISTPoison
from models import BadNet

parser = argparse.ArgumentParser(description='Randomly display clean and attacked images with predictions.')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--batch_size', type=int, default=10, help='Number of images to display (default: 10)')
parser.add_argument('--device', default='cuda:0', help='device to use for training / testing (cpu, or cuda:0, default: cpu)')
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning rate for MNISTPoison dataset')
parser.add_argument('--trigger_label', type=int, default=1, help='Trigger target label')
parser.add_argument('--trigger_size', type=int, default=5, help='Size of the trigger square')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Path to trigger image (optional)')
args = parser.parse_args()

def test_clean_images(data_loader, model, device):
    """Test the model on clean images."""
    print("Testing model on clean images...")
    model.eval()
    to_pil = ToPILImage()

    for clean_images, labels in data_loader:
        clean_images, labels = clean_images.to(device), labels.to(device)

        # Model predictions
        clean_outputs = model(clean_images)
        _, clean_preds = torch.max(clean_outputs, 1)

        # Display clean images with predictions
        fig, axes = plt.subplots(1, args.batch_size, figsize=(15, 3))

        for i in range(args.batch_size):
            img_clean = clean_images[i].cpu().squeeze(0).numpy()
            axes[i].imshow(img_clean, cmap="gray")
            axes[i].set_title(f"Clean Pred: {clean_preds[i].item()}")
            axes[i].axis("off")

        plt.suptitle("Clean Images with Predictions")
        plt.show()
        break

def test_triggered_images(data_loader, model, device, trigger_handler):
    """Test the model on images with triggers added manually."""
    print("Testing model on triggered images...")
    model.eval()
    to_pil = ToPILImage()
    to_tensor = ToTensor()

    for clean_images, labels in data_loader:
        clean_images, labels = clean_images.to(device), labels.to(device)
        triggered_images = clean_images.clone()

        # Add triggers to images manually
        for i in range(triggered_images.size(0)):
            img_pil = to_pil(triggered_images[i].cpu())  # Convert to PIL
            img_pil = trigger_handler.put_trigger(img_pil)  # Add trigger
            triggered_images[i] = to_tensor(img_pil).to(device)  # Back to tensor

        # Model predictions
        triggered_outputs = model(triggered_images)
        _, triggered_preds = torch.max(triggered_outputs, 1)

        # Display triggered images with predictions
        fig, axes = plt.subplots(1, args.batch_size, figsize=(15, 3))

        for i in range(args.batch_size):
            img_triggered = triggered_images[i].cpu().squeeze(0).numpy()
            axes[i].imshow(img_triggered, cmap="gray")
            axes[i].set_title(f"Trigger Pred: {triggered_preds[i].item()}")
            axes[i].axis("off")

        plt.suptitle("Triggered Images with Predictions")
        plt.show()
        break

def main():
    # Setup GPU device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load clean MNIST dataset manually without pollution
    print("Loading clean MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    from torchvision.datasets import MNIST  # Use original clean MNIST dataset
    
    clean_dataset = MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )
    data_loader_clean = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=True)

    # Load model
    model = BadNet(input_channels=1, output_num=10).to(device)
    model_path = "./checkpoints/badnet-MNIST.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))

    # Test on clean images
    test_clean_images(data_loader_clean, model, device)

    # Test on triggered images
    trigger_handler = clean_dataset.trigger_handler  # Use original trigger handler
    test_triggered_images(data_loader_clean, model, device, trigger_handler)
if __name__ == "__main__":
    main()
    