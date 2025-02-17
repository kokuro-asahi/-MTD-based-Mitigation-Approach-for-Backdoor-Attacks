import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from models import BadNet

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
        # 使用 register_full_backward_hook（如果可用）替代 register_backward_hook
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(self.save_gradients)
        else:
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
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 防止除零
        return cam

def load_model():
    model = BadNet(input_channels=1, output_num=10)
    # 修改这里，设置 weights_only=True 以避免 pickle 警告
    model.load_state_dict(torch.load("./checkpoints/badnet-MNIST.pth", weights_only=True))
    model.eval()
    return model

def generate_custom_images():
    """
    生成 49 张定制图片，每张图片为 28x28，
    图片中只有一个 4x4 的白色区域，其位置由 7x7 网格决定。
    """
    images = []
    for i in range(7):       # 行
        for j in range(7):   # 列
            img_array = np.zeros((28, 28), dtype=np.uint8)
            # 将对应的 4x4 区域设为白色
            img_array[i*4:(i+1)*4, j*4:(j+1)*4] = 255
            img = Image.fromarray(img_array, mode='L')
            images.append(img)
    return images

if __name__ == "__main__":
    dataset = "MNIST"
    transform, detransform = build_transform(dataset)

    custom_images = generate_custom_images()
    model = load_model()
    grad_cam = GradCAM(model, model.conv1[0])

    # 存放预测结果与对应的 GradCAM 热图
    predictions = []
    heatmaps = []

    # 遍历 49 张图片，依次得到预测及热图
    for img in custom_images:
        # 预处理：将 PIL 图像转换为 tensor
        img_tensor = transform(img).unsqueeze(0).to("cpu")
        
        # 模型预测
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
        predictions.append(pred)
        
        # 生成 GradCAM 热图，使用预测的类别作为目标类别
        heatmap = grad_cam.generate(img_tensor, target_class=pred)
        heatmaps.append(heatmap)
    
    # 可视化：将原始图片与热图叠加显示
    fig, axes = plt.subplots(7, 7, figsize=(12, 12))
    for idx, (img, hm, pred) in enumerate(zip(custom_images, heatmaps, predictions)):
        ax = axes[idx // 7, idx % 7]
        # 将热图转换为 28x28 大小，便于与原图叠加
        hm_img = Image.fromarray(np.uint8(255 * hm)).resize((28, 28), Image.Resampling.LANCZOS)
        hm_np = np.array(hm_img)
        
        # 将原始图像转换为 numpy 数组
        img_np = np.array(img)
        # 简单线性加权叠加
        overlay = (img_np.astype(np.float32) * 0.5 + hm_np.astype(np.float32) * 0.5)
        overlay = np.uint8(overlay / overlay.max() * 255) if overlay.max() > 0 else overlay
        
        ax.imshow(overlay, cmap='jet')
        ax.set_title(f'Pred: {pred}', fontsize=8)
        ax.axis('off')
    
    plt.suptitle("", fontsize=16)
    plt.tight_layout()
    plt.show()
