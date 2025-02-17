import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from models import BadNet  # 假设你已有的模型
import pandas as pd
# 已有的函数定义（build_transform, GradCAM, load_model等）
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

        # 计算每个通道的权重
        weights = torch.mean(self.gradients, dim=(2, 3))
        cam = torch.sum(weights[:, :, None, None] * self.features, dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 防止除0
        return cam

    def generate_all_classes_average(self, input_image):
        all_heatmaps = []
        for target_class in range(10):
            output = self.model(input_image)
            self.model.zero_grad()

            target = torch.zeros_like(output)
            target[0][target_class] = 1
            output.backward(gradient=target, retain_graph=True)

            # 计算权重并生成热力图
            weights = torch.mean(self.gradients, dim=(2, 3))
            cam = torch.sum(weights[:, :, None, None] * self.features, dim=1)
            cam = F.relu(cam)
            cam = cam.squeeze().detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            all_heatmaps.append(cam)    
        # 对 10 个类别的热力图取平均
        return np.mean(all_heatmaps, axis=0)

def load_model():
    model = BadNet(input_channels=1, output_num=10)
    model.load_state_dict(torch.load("./checkpoints/badnet-MNIST.pth"))
    #model.load_state_dict(torch.load("./checkpoints/badnet_mnist_0,0_5.pth"))
    #model.load_state_dict(torch.load("./checkpoints/badnet_mnist_12,12_5.pth"))
    #model.load_state_dict(torch.load("./checkpoints/badnet_mnist_untriggered.pth"))
    model.eval()
    return model


def generate_custom_images(image_size=28, white_block_size=4, grid_rows=None, grid_cols=None):
    
    
    if grid_rows is None:
        grid_rows = image_size // white_block_size
    if grid_cols is None:
        grid_cols = image_size // white_block_size

    images = []
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            
            img_array = np.full((image_size, image_size), 10, dtype=np.uint8)
           
            start_i = i * white_block_size
            start_j = j * white_block_size
            
            if start_i + white_block_size <= image_size and start_j + white_block_size <= image_size:
                img_array[start_i:start_i+white_block_size, start_j:start_j+white_block_size] = 255
            else:
                
                raise ValueError("参数不匹配：白块超出图片边界，请调整 image_size、white_block_size、grid_rows 和 grid_cols")
            
            img = Image.fromarray(img_array, mode='L')
            images.append(img)
    return images

if __name__ == "__main__":
    dataset = "MNIST"
    transform, _ = build_transform(dataset)
    
    
    image_size = 28         
    white_block_size = 2 # 白点大小
    grid_rows =  14    # 行数
    grid_cols = 14   #列数
    

    custom_images = generate_custom_images(image_size=image_size,
                                           white_block_size=white_block_size,
                                           grid_rows=grid_rows,
                                           grid_cols=grid_cols)
    
    model = load_model()
    grad_cam = GradCAM(model, model.conv1[0])
    
    
    n = 1  
    
    
    accumulated_attention = np.zeros((image_size, image_size), dtype=np.float32)
    
    
    for rep in range(n):
        total_attention = np.zeros((image_size, image_size), dtype=np.float32)
        for img in custom_images:
            
            img_tensor = transform(img).unsqueeze(0).to("cpu")
            
            
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
            
            #使用预测类别
            #heatmap = grad_cam.generate(img_tensor, target_class=pred)
            #使用十类平均
            heatmap = grad_cam.generate_all_classes_average(img_tensor)    
            
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_img = Image.fromarray(heatmap_uint8)
            heatmap_resized_img = heatmap_img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            heatmap_resized = np.array(heatmap_resized_img) / 255.0
            
           
            total_attention += heatmap_resized
        
        accumulated_attention += total_attention

    
    average_attention = accumulated_attention / n

    
    plt.figure(figsize=(5, 5))
    plt.imshow(average_attention, cmap='jet')
    plt.title(f"{image_size}x{image_size}  repeat {n} ")
    plt.colorbar()
    plt.axis('off')
    plt.show()

    
    np.savetxt("average_attention.csv", average_attention, delimiter=",")
    plt.imsave("average_attention.png", average_attention, cmap="jet")