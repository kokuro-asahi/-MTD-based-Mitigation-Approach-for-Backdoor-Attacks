import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import BadNet  

def log_weights_to_tensorboard(model, writer, step):
    
    for name, param in model.named_parameters():
        if 'weight' in name:  
            writer.add_histogram(f"Weights/{name}", param.cpu().data.numpy(), step)
            
           
            if len(param.shape) == 4:  
               
                kernels = param[:, 0, :, :] 
                kernels = kernels.unsqueeze(1)  
                writer.add_images(f"Conv_Weights/{name}", kernels, step)

if __name__ == "__main__":
   
    model_path = "badnet-MNIST.pth"  
    log_dir = "runs/weight_visualization" 

    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

   
    print("Defining model...")
    model = BadNet(input_channels=1, output_num=10) 

    
    print(f"Loading model weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))  
    model.load_state_dict(state_dict) 
    model.eval()  

    
    writer = SummaryWriter(log_dir=log_dir)

   
    log_weights_to_tensorboard(model, writer, step=0)

   
    writer.close()
    

    
    writer = SummaryWriter(log_dir=log_dir)

    
    log_weights_to_tensorboard(model, writer, step=0)
    writer.close()

    
    print(f"运行以下命令启动 TensorBoard：")
    print(f"tensorboard --logdir={os.path.abspath(log_dir)}")
