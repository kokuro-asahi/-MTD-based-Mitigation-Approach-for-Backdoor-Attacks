import argparse
import os
import pathlib
import re

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import MNISTPoison
from models import BadNet
from torchvision.transforms import ToPILImage, ToTensor
parser = argparse.ArgumentParser(description='Randomly display clean and attacked images with predictions.')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--batch_size', type=int, default=10, help='Number of images to display (default: 10)')
parser.add_argument('--device', default='cuda:0', help='device to use for training / testing (cpu, or cuda:0, default: cpu)')
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning rate for MNISTPoison dataset')
parser.add_argument('--trigger_label', type=int, default=1, help='Trigger target label')
parser.add_argument('--trigger_size', type=int, default=5, help='Size of the trigger square')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Path to trigger image (optional)')
args = parser.parse_args()

def main():
    # Setup GPU device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load clean MNIST dataset first
    print("Loading clean MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    clean_dataset = MNISTPoison(
        args=args,
        root="./data",
        train=False,
        transform=transform,
        download=True
    )
    clean_dataset.poi_indices = []  # Ensure no data is poisoned yet
    data_loader_clean = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=True)

    # Load model
    model = BadNet(input_channels=1, output_num=10).to(device)
    model_path = "./checkpoints/badnet-MNIST.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Display clean and triggered images with predictions
    print("Displaying clean images and triggered images with predictions...")
    to_pil = ToPILImage()
    to_tensor = ToTensor()

    for clean_images, labels in data_loader_clean:
        clean_images, labels = clean_images.to(device), labels.to(device)

        # Clone clean images and add triggers manually
        triggered_images = clean_images.clone()
        for i in range(triggered_images.size(0)):
            img_pil = to_pil(triggered_images[i].cpu())  # Convert to PIL
            img_pil = clean_dataset.trigger_handler.put_trigger(img_pil)  # Add trigger
            triggered_images[i] = to_tensor(img_pil).to(device)  # Back to tensor

        # Model predictions
        clean_outputs = model(clean_images)
        triggered_outputs = model(triggered_images)
        _, clean_preds = torch.max(clean_outputs, 1)
        _, triggered_preds = torch.max(triggered_outputs, 1)

        # Display clean and triggered images
        fig, axes = plt.subplots(2, args.batch_size, figsize=(15, 6))

        for i in range(args.batch_size):
            # Original clean image with prediction
            img_clean = clean_images[i].cpu().squeeze(0).numpy()
            axes[0, i].imshow(img_clean, cmap="gray")
            axes[0, i].set_title(f"Clean Pred: {clean_preds[i].item()}")
            axes[0, i].axis("off")

            # Triggered image with prediction
            img_triggered = triggered_images[i].cpu().squeeze(0).numpy()
            axes[1, i].imshow(img_triggered, cmap="gray")
            axes[1, i].set_title(f"Trigger Pred: {triggered_preds[i].item()}")
            axes[1, i].axis("off")

        plt.suptitle("Top: Clean Images with Predictions | Bottom: Triggered Images with Predictions")
        plt.show()
        break  # Display one batch only

if __name__ == "__main__":
    main()