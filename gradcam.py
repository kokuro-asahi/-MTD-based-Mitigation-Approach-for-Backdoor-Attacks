import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataset.poisoned_dataset import CIFAR10Poison, MNISTPoison
from models import BadNet
from torchvision.datasets import CIFAR10, MNIST
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def build_transform(dataset):
    if dataset == "CIFAR10":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "MNIST":
        mean, std = (0.5,), (0.5,)
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    
    return transform, detransform



class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

        
        self.target_layer.register_forward_hook(self.save_features)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.features = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, target_class):
        
        output = self.model(input_image)
        
        
        self.model.zero_grad()
        target = torch.zeros_like(output)
        target[0][target_class] = 1
        output.backward(gradient=target, retain_graph=True)

        
        weights = torch.mean(self.gradients, dim=(2, 3)) 
        cam = torch.sum(weights[:, :, None, None] * self.features, dim=1)  
        cam = F.relu(cam)  

        
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def load_model():
    model = BadNet(input_channels=1, output_num=10) 
    model.load_state_dict(torch.load("./checkpoints/badnet-MNIST.pth")) 
    #model.load_state_dict(torch.load("./checkpoints/badnet_mnist_untriggered.pth"))
    model.eval()
    return model

def preprocess_image(image, dataset):
    transform, _ = build_transform(dataset)
    
    if isinstance(image, torch.Tensor):
        return transform.transforms[-1](image).unsqueeze(0) 
    return transform(image).unsqueeze(0)

def visualize_combined(
    original_image, triggered_image, original_heatmap1, triggered_heatmap1, original_heatmap2, triggered_heatmap2,dataset, 
    original_pred, triggered_pred
):
    original_image_np = np.array(original_image)
    if original_image_np.shape[0] == 1:  
        original_image_np = original_image_np.squeeze(0)

    triggered_image_np = np.array(triggered_image.squeeze(0))
    if triggered_image_np.shape[0] == 1:
        triggered_image_np = triggered_image_np.squeeze(0)

    original_heatmap1 = np.uint8(255 * original_heatmap1)
    original_heatmap1 = Image.fromarray(original_heatmap1).resize(original_image_np.shape[::-1], Image.Resampling.LANCZOS)
    original_heatmap1 = np.array(original_heatmap1)

    original_heatmap2 = np.uint8(255 * original_heatmap2)
    original_heatmap2 = Image.fromarray(original_heatmap2).resize(original_image_np.shape[::-1], Image.Resampling.LANCZOS)
    original_heatmap2 = np.array(original_heatmap2)

    triggered_heatmap1 = np.uint8(255 * triggered_heatmap1)
    triggered_heatmap1 = Image.fromarray(triggered_heatmap1).resize(triggered_image_np.shape[::-1], Image.Resampling.LANCZOS)
    triggered_heatmap1 = np.array(triggered_heatmap1)

    triggered_heatmap2 = np.uint8(255 * triggered_heatmap2)
    triggered_heatmap2 = Image.fromarray(triggered_heatmap2).resize(triggered_image_np.shape[::-1], Image.Resampling.LANCZOS)
    triggered_heatmap2 = np.array(triggered_heatmap2)


    
    original_superimposed1 = original_image_np * 0.5 + original_heatmap1 * 0.5
    triggered_superimposed1 = triggered_image_np * 0.5 + triggered_heatmap1 * 0.5

    original_superimposed2 = original_image_np * 0.5 + original_heatmap2 * 0.5
    triggered_superimposed2 = triggered_image_np * 0.5 + triggered_heatmap2 * 0.5



    original_superimposed1 = original_superimposed1 / original_superimposed1.max() * 255
    triggered_superimposed1 = triggered_superimposed1 / triggered_superimposed1.max() * 255

    original_superimposed2 = original_superimposed2 / original_superimposed2.max() * 255
    triggered_superimposed2 = triggered_superimposed2 / triggered_superimposed2.max() * 255



    original_superimposed1 = original_superimposed1.astype(np.uint8)
    triggered_superimposed1 = triggered_superimposed1.astype(np.uint8)

    original_superimposed2 = original_superimposed2.astype(np.uint8)
    triggered_superimposed2 = triggered_superimposed2.astype(np.uint8)

    
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title(f'Original Image (Pred: {original_pred})')
    if dataset == "MNIST":
        plt.imshow(original_image_np, cmap='gray')
    else:
        plt.imshow(original_image_np)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Original Heatmap conv1')
    plt.imshow(original_superimposed1, cmap='jet')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Original Heatmap conv2')
    plt.imshow(original_superimposed2, cmap='jet')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title(f'Triggered Image (Pred: {triggered_pred})')
    if dataset == "MNIST":
        plt.imshow(triggered_image_np, cmap='gray')
    else:
        plt.imshow(triggered_image_np)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Triggered Heatmap')
    plt.imshow(triggered_superimposed1, cmap='jet')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Triggered Heatmap')
    plt.imshow(triggered_superimposed2, cmap='jet')
    plt.axis('off')

    plt.show()
if __name__ == "__main__":
    dataset = "MNIST"  

    
    if dataset == "MNIST":
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=build_transform(dataset)[0])
    elif dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=build_transform(dataset)[0])

    
    label_image_count = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1

    }

    selected_images = []
    selected_labels = []

    
    label_counter = {label: 0 for label in label_image_count.keys()}
    for img, lbl in train_dataset:
        if lbl in label_image_count and label_counter[lbl] < label_image_count[lbl]:
            selected_images.append(img)
            selected_labels.append(lbl)
            label_counter[lbl] += 1

        if all(label_counter[label] >= label_image_count[label] for label in label_image_count):
            break

    for i, (image, label) in enumerate(zip(selected_images, selected_labels)):
        
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        pil_img = to_pil(image)  
        trigger_img = Image.open("./triggers/trigger_white.png").convert("RGB")
        trigger_img = trigger_img.resize((5, 5))  
        pil_img.paste(trigger_img, (23, 23)) 

        new_img_triggered = to_tensor(pil_img).unsqueeze(0).to("cpu") 

        
        model = load_model()
        
        grad_cam1 = GradCAM(model, model.conv1[0])
        grad_cam2 = GradCAM(model, model.conv2[0])
        
        with torch.no_grad():
            original_output = model(image.unsqueeze(0).to("cpu"))
            original_pred = torch.argmax(original_output, dim=1).item()

            triggered_output = model(new_img_triggered)
            triggered_pred = torch.argmax(triggered_output, dim=1).item()

        
        original_heatmap1 = grad_cam1.generate(image.unsqueeze(0).to("cpu"), target_class=label)
        triggered_heatmap1 = grad_cam1.generate(new_img_triggered, target_class=label)

        original_heatmap2 = grad_cam2.generate(image.unsqueeze(0).to("cpu"), target_class=label)
        triggered_heatmap2 = grad_cam2.generate(new_img_triggered, target_class=label)

        
        visualize_combined(
            image, new_img_triggered, original_heatmap1, triggered_heatmap1, original_heatmap2, triggered_heatmap2,dataset, 
            original_pred, triggered_pred
        )



