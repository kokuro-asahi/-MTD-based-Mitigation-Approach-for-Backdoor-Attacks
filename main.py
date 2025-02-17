import argparse
import os
import pathlib
import re
import time
import datetime
import torch.nn.functional as F
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_poisoned_training_set, build_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch
from models import BadNet

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--print_graph', action='store_true')

parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=50, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers to load data, default: 0')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cuda:0', help='device to use for training / testing (cpu, or cuda:0, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
args = parser.parse_args()


def masl_lower_zero_hook(module, input, output):
    output[:, :, -output.size(2) // 2 :, -output.size(3) // 2 :] = 0
    return output

def mask_lower_half_hook(module, input, output):
    output[:, :, -output.size(2) // 2 :, -output.size(3) // 2 :] = 0.5
    return output

def mask_with_blur_hook_conv2(module, input, output):
    """
    针对 conv2 的模糊 Hook。
    """
    blur_region = output[:, :, -output.size(2) // 2 :, -output.size(3) // 2 :]

    if blur_region.size(2) < 5 or blur_region.size(3) < 5:
        return output

    # 定义高斯核
    gaussian_kernel = torch.tensor([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ], dtype=torch.float32)
    gaussian_kernel /= gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, 5, 5).to(output.device)  # 高斯核大小为 5x5

    
    blurred = torch.stack([
        F.conv2d(blur_region[:, i:i+1], gaussian_kernel, padding=2)  # 单通道卷积
        for i in range(blur_region.size(1))  # 遍历通道
    ], dim=1)

    
    weight_factor = 0.1
    blur_region = blurred * weight_factor

    
    output[:, :, -output.size(2) // 2 :, -output.size(3) // 2 :] = blur_region
    return output

def mask_with_blur_hook_conv1(module, input, output):
    """
    针对 conv1 。
    """
    
    blur_region = output[:, :, -output.size(2) // 2 :, -output.size(3) // 2 :]

    
    if blur_region.size(2) < 2 or blur_region.size(3) < 2:
        return output  

    
    blur_region = F.avg_pool2d(blur_region, kernel_size=3, stride=2, padding=1)

    
    if blur_region.shape != output[:, :, -output.size(2) // 2 :, -output.size(3) // 2 :].shape:
        blur_region = F.interpolate(
            blur_region,
            size=output[:, :, -output.size(2) // 2 :, -output.size(3) // 2 :].shape[2:],
            mode="nearest"
        )

    
    output[:, :, -output.size(2) // 2 :, -output.size(3) // 2 :] = blur_region
    return output

def mask_with_mean_hook(module, input, output):
    mean_value = torch.mean(output, dim=(2, 3), keepdim=True)
    output[:, :, -output.size(2) // 2 :, -output.size(3) // 2 :] = mean_value
    return output


processlist = [
    
    masl_lower_zero_hook,
    mask_lower_half_hook, 
    mask_with_mean_hook,
    mask_with_blur_hook_conv2
]


def register_partial_mask_hook(layer, hook_function, region):
    
    #仅对特定区域应用屏蔽操作。

    
    def partial_mask_hook(module, input, output):
        h, w = output.size(2), output.size(3)
        h_mid, w_mid = h // 2, w // 2
        if region == "upper_left":
            output[:, :, :h_mid, :w_mid] = hook_function(module, input, output[:, :, :h_mid, :w_mid])
        elif region == "upper_right":
            output[:, :, :h_mid, w_mid:] = hook_function(module, input, output[:, :, :h_mid, w_mid:])
        elif region == "lower_left":
            output[:, :, h_mid:, :w_mid] = hook_function(module, input, output[:, :, h_mid:, :w_mid])
        elif region == "lower_right":
            output[:, :, h_mid:, w_mid:] = hook_function(module, input, output[:, :, h_mid:, w_mid:])
        return output

    layer.register_forward_hook(partial_mask_hook)

def main():
    print("{}".format(args).replace(', ', ',\n'))

    # Setup GPU device
    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Create related paths
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("\n# Load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)

    data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)

    #basic_model_path = "./checkpoints/badnet-%s.pth" % args.dataset
    #basic_model_path = "./checkpoints/badnet_mnist_untriggered.pth"
    #basic_model_path = "./checkpoints/badnet_mnist_0,0_5.pth"  
    basic_model_path = "./checkpoints/badnet_mnist_12,12_5.pth" 
    start_time = time.time()
    if args.load_local:
        
    ######屏蔽 
        '''
        print("## Load model from : %s" % basic_model_path)
        model.load_state_dict(torch.load(basic_model_path), strict=True)
        model.conv2.register_forward_hook(mask_with_mean_hook)

        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device,args)
        print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")
        '''
        log_results = []
        regions = ["left_right"]
        layers = [ "conv2"]
        for layer_name in layers:
            for hook_function in processlist:
                for region in regions:
                    
                    model.load_state_dict(torch.load(basic_model_path), strict=True)
                    torch.cuda.empty_cache()
                    
                    layer = getattr(model, layer_name)[2] 
                    handle = layer.register_forward_hook(
                        lambda module, input, output: register_partial_mask_hook(layer, hook_function, region)
                    )
                    
                    

                    
                    test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device, args)

                    handle.remove()
                    
                    result = {
                        "layer": layer_name,
                        "hook_function": hook_function.__name__,
                        "region": region,
                        "TCA": test_stats["clean_acc"],
                        "ASR": test_stats["asr"],
                    }
                    log_results.append(result)

                    del test_stats
                    torch.cuda.empty_cache()
                
                #print(f"Layer: {layer_name}, Hook: {hook_function.__name__}, Region: {region}")
                #print(f"Test Clean Accuracy (TCA): {test_stats['clean_acc']:.4f}")
                #print(f"Attack Success Rate (ASR): {test_stats['asr']:.4f}")
                
        log_df = pd.DataFrame(log_results)
        directory = "./experiment_results_log"
        file_name = "conv2[2]_left_right.csv"
        log_file_path = os.path.join(directory, file_name)
        os.makedirs(directory, exist_ok=True)
        
        log_df.to_csv(log_file_path, index=False)
        print(f"experiment results save in  {log_file_path}")
    else:
        print(f"Start training for {args.epochs} epochs")
        stats = []
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, args.loss, device)
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model,device, args)
            print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")

            # Save model
            torch.save(model.state_dict(), basic_model_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
            }

            # Save training stats
            stats.append(log_stats)
            df = pd.DataFrame(stats)
            df.to_csv("./logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()

