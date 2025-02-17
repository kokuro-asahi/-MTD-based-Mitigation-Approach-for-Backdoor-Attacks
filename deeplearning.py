import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from torchvision import transforms
from PIL import Image
def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = torch.optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = torch.optim.Adam(param, lr=lr)
    return optimizer


def train_one_epoch(data_loader, model, criterion, optimizer, loss_mode, device):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x
        
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss
    return {
            "loss": running_loss.item() / len(data_loader),
            }

def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device,args):
    ta = eval(data_loader_val_clean, model, device, print_perform=True)
    asr = eval(data_loader_val_poisoned, model, device, print_perform=False)

# 删除
    if args.print_graph:
        sample_per_label = {}
        class_names = data_loader_val_clean.dataset.classes  

        # 删除
        for (batch_x, batch_y) in data_loader_val_clean:

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            batch_y_predict = model(batch_x)
            batch_y_predict = torch.argmax(batch_y_predict, dim=1)

            for i in range(len(batch_y)):
                label_id = batch_y[i].item()
                if label_id not in sample_per_label:
                    # 保存 (图像, 预测值)
                    sample_per_label[label_id] = (
                        batch_x[i].detach().cpu(),   # 将图像搬到 CPU，方便后续可视化或处理
                        batch_y_predict[i].item()
                    )
                    # 如果已经收集到所有 label，就可以提前结束
                    if len(sample_per_label) == len(class_names):
                        break
        
        printgraph(sample_per_label, class_names)
        
        for label_id, (img_tensor, pred_label) in sample_per_label.items():
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            # 1. Tensor -> PIL
            pil_img = to_pil(img_tensor)  # img_tensor shape: (C, H, W)

            # 2. 在 PIL 图像上粘贴 trigger_img
            #    paste() 是就地(in-place)操作，不会返回新对象
            trigger_img = Image.open(args.trigger_path).convert('RGB')
            pil_img.paste(trigger_img, (28 - args.trigger_size, 28 - args.trigger_size))

            # 3. PIL -> Tensor
            #    得到新的张量 (C, H, W)
            new_img_tensor = to_tensor(pil_img)

            new_img_tensor = new_img_tensor.unsqueeze(0).to(device)  # 增加 batch 维度 (1, C, H, W)
            
            with torch.no_grad():
                y_pred = model(new_img_tensor)  # 前向传播
                new_pred_label = torch.argmax(y_pred, dim=1).item()  # 获取预测类别
            new_img_tensor_cpu = new_img_tensor.squeeze(0).cpu()
            sample_per_label[label_id] = (new_img_tensor_cpu, new_pred_label)   
        
        printgraph(sample_per_label, class_names)    




    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            }

def eval(data_loader, model, device, batch_size=64, print_perform=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    
    for (batch_x, batch_y) in tqdm(data_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_y_predict = model(batch_x)
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())
        

        
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))
    
    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }

def printgraph(sample_dict, class_names, ncols=4):
    """
    接收一个字典 sample_dict，格式例如：
        {
            label_id: (image_tensor, predicted_label),
            ...
        }
    以及 class_names，表示标签名称列表。
    
    一次性将所有图片显示在同一个 Figure 中，避免每张图都单独弹窗。
    - ncols: 每行放多少张图（可根据需要调整）。
    """
    sample_dict = {k:sample_dict[k] for k in sorted(sample_dict)}
    n_samples = len(sample_dict)
    if n_samples == 0:
        print("No images to display!")
        return

    # 根据 ncols 计算需要多少行
    nrows = math.ceil(n_samples / ncols)
    
    # 创建子图网格
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=(3 * ncols, 3 * nrows))

    # 如果只有一行，axes 可能不是二维数组，这里统一做扁平化处理
    # 这样后面就可以用 axes[i] 索引
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    else:
        # axes 是 2D list，需要 flatten 成一维
        axes = axes.flatten()

    # 将 sample_dict 的 key-value 转成一个可枚举列表方便遍历
    sample_items = list(sample_dict.items())  # [(label_id, (img_tensor, pred)), ...]
    
    for i, (label_id, (img_tensor, pred_label)) in enumerate(sample_items):
        # 把 (C, H, W) 转为 (H, W, C)
        img_numpy = img_tensor.permute(1, 2, 0).numpy()
        
        # 如果图像有标准化，需要在此处进行 “反标准化”
        # 这里只是演示直接显示
        cmap = 'gray' if img_numpy.shape[-1] == 1 else None
        
        # 获取真实标签名、预测标签名
        true_label_name = class_names[label_id] if label_id < len(class_names) else str(label_id)
        pred_label_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
        
        axes[i].imshow(img_numpy, cmap=cmap)
        axes[i].set_title(f"True: {true_label_name}\nPred: {pred_label_name}")
        axes[i].axis('off')
    
    # 如果子图数量大于图像数量，隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
