import torch
print(torch.cuda.is_available())  # 检查是否检测到 GPU
print(torch.cuda.device_count())  # 检查可用 GPU 数量
print(torch.cuda.current_device())  # 检查当前设备 ID
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # 检查 GPU 名称
