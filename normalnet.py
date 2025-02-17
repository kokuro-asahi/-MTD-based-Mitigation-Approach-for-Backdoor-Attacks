import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import BadNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    
    model = BadNet(input_channels=1, output_num=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
    def train(model, train_loader, criterion, optimizer, device, epochs=10):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")
    
    def test(model, test_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
   
    train(model, train_loader, criterion, optimizer, device, epochs=10)
    test(model, test_loader, device)
    
    
    torch.save(model.state_dict(), "./checkpoints/badnet_mnist_untriggered.pth")
    print("Model saved to ./checkpoints/badnet_mnist_untriggered.pth")


if __name__ == "__main__":
    main()